"""
The horse race.

Runs several well-documented strategies on the same data, with the same costs
and the same benchmark, and reports them side by side. The point is not to
find "the best strategy". It is to stop arguing about which one to use and
look at what each one actually did.

The strategies here were not invented for this project. Each one has decades
of published work behind it, which matters: a rule someone made up last week
and backtested until it looked good tells you nothing, because it was fitted
to the same history you are testing it on.

  Buy and hold          The benchmark. Any strategy that cannot beat this,
                        after costs, is not worth running.

  200-day trend filter  Hold the index while it is above its 200-day average,
                        hold short-term bonds when it is below. Meb Faber's
                        2007 paper is the standard reference. It historically
                        did not raise returns much. It cut drawdowns hard.

  Dual momentum         Each month, hold whichever of US stocks or
                        international stocks did better over the past year,
                        but only if that beat bonds. Otherwise hold bonds.
                        Gary Antonacci's formulation.

  Cross-sectional       Each month, hold the strongest handful of names from
  momentum              the watchlist by 6-month return, skipping the most
                        recent month. Jegadeesh and Titman, 1993. This has
                        the most persistent published evidence of any simple
                        rule, and it also has brutal crash risk.

  Equal weight          Hold the whole watchlist equally, rebalanced monthly.
                        A control, to show how much of any result is just
                        "owned stocks during a bull market".

  Trend pullback        Your current rules, run through the trade engine.

An honest warning about reading the output: the strategy at the top of the
table is partly the luckiest one, not purely the best one. Testing six
strategies means the winner beat five others by chance as well as by merit.
That is why the out-of-sample column exists, and why you should weight it far
more heavily than the full-period column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# Assets the allocation strategies need, on top of your stock watchlist.
BENCHMARK = "SPY"
INTERNATIONAL = "EFA"     # developed markets outside the US
BONDS = "AGG"             # US aggregate bond market
CASH = "SHY"              # 1-3 year treasuries, the "safe" sleeve

EXTRA_ASSETS = [BENCHMARK, INTERNATIONAL, BONDS, CASH]

# A universe that does not cheat.
#
# Running momentum over a watchlist of today's mega caps is not a test, it is
# a memory. Whoever wrote the list already knows which companies won, so the
# backtest inherits that knowledge and reports it as skill. The tell is that
# holding the whole list equally, with no strategy at all, also beats the
# index by a mile.
#
# These are broad index and sector funds that all existed and were freely
# buyable in 2005. Nobody had to know anything about the future to pick this
# list, so a momentum result on it means something.
HONEST_UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV",
                   "SMH", INTERNATIONAL, BONDS]

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def close_matrix(bars: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One column of closing prices per symbol, on a shared calendar."""
    cols = {s: df["close"] for s, df in bars.items()}
    px = pd.DataFrame(cols).sort_index()
    return px.dropna(how="all")


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).last().values)


def simulate(prices: pd.DataFrame, weights: pd.DataFrame,
             slippage_bps: float = 5.0,
             start_equity: float = 10_000.0) -> pd.DataFrame:
    """Turn a table of target weights into an equity curve.

    The weight row for day t is what you decided at the CLOSE of day t, so it
    earns day t+1's return, not day t's. That one-line shift is the difference
    between a real backtest and a machine that buys yesterday's winners with
    yesterday's knowledge.
    """
    prices = prices.sort_index()
    rets = prices.pct_change().fillna(0.0)

    w = weights.reindex(prices.index).ffill().fillna(0.0)
    w = w.reindex(columns=prices.columns).fillna(0.0)

    held = w.shift(1).fillna(0.0)               # yesterday's decision, today's return
    gross = (held * rets).sum(axis=1)

    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (slippage_bps / 10_000.0)

    net = gross - cost
    equity = start_equity * (1.0 + net).cumprod()

    return pd.DataFrame({
        "equity": equity,
        "ret": net,
        "exposure": held.sum(axis=1),
        "turnover": turnover,
    })


def metrics(sim: pd.DataFrame, label: str = "") -> dict:
    eq = sim["equity"]
    if len(eq) < 2:
        return {"strategy": label, "trades": 0}

    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

    dd = (eq.cummax() - eq) / eq.cummax()
    max_dd = dd.max()

    r = sim["ret"]
    sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan
    downside = r[r < 0].std()
    sortino = r.mean() / downside * np.sqrt(TRADING_DAYS) if downside and downside > 0 else np.nan
    calmar = cagr / max_dd if max_dd > 0 else np.nan

    # Group by calendar period rather than resample("ME"). The "ME" alias only
    # exists in pandas 2.2 and later, and this has to run on whatever version
    # the machine happens to have.
    monthly = eq.groupby(eq.index.to_period("M")).last().pct_change().dropna()
    yearly = eq.groupby(eq.index.to_period("Y")).last().pct_change().dropna()

    return {
        "strategy": label,
        "total_pct": total * 100,
        "cagr_pct": cagr * 100,
        "max_dd_pct": max_dd * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "worst_year_pct": yearly.min() * 100 if len(yearly) else np.nan,
        "best_year_pct": yearly.max() * 100 if len(yearly) else np.nan,
        "pct_months_up": (monthly > 0).mean() * 100 if len(monthly) else np.nan,
        "exposure_pct": sim["exposure"].mean() * 100,
        "turnover_yr": sim["turnover"].sum() / years if years > 0 else np.nan,
        "final": eq.iloc[-1],
    }


# ---------------------------------------------------------------------------
# Strategies. Each returns a weights frame indexed by date.
# ---------------------------------------------------------------------------

def w_buy_hold(prices: pd.DataFrame, symbol: str = BENCHMARK) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if symbol in w.columns:
        w[symbol] = 1.0
    return w


def _blank(prices: pd.DataFrame) -> pd.DataFrame:
    """All NaN, so that only the rows we set survive into the forward fill.

    Rebalancing happens on month ends. Every other row inherits the previous
    decision, which is what "hold until the next rebalance" means.
    """
    return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)


def _settle(w: pd.DataFrame) -> pd.DataFrame:
    return w.ffill().fillna(0.0)


def w_equal_weight(prices: pd.DataFrame, universe: List[str]) -> pd.DataFrame:
    cols = [s for s in universe if s in prices.columns]
    w = _blank(prices)
    if not cols:
        return w.fillna(0.0)
    for d in month_ends(prices.index):
        live = [c for c in cols if not np.isnan(prices.at[d, c])]
        if not live:
            continue
        w.loc[d] = 0.0
        w.loc[d, live] = 1.0 / len(live)
    return _settle(w)


def w_trend_filter(prices: pd.DataFrame, symbol: str = BENCHMARK,
                   safe: str = CASH, ma: int = 200) -> pd.DataFrame:
    """In the index when it is above its long average, in short bonds when not.

    Checked at each month end, which is how the published version works.
    Checking daily produces more whipsaw and more cost for no benefit.
    """
    w = _blank(prices)
    if symbol not in prices.columns:
        return w.fillna(0.0)
    avg = prices[symbol].rolling(ma, min_periods=ma).mean()

    for d in month_ends(prices.index):
        a, p = avg.get(d, np.nan), prices[symbol].get(d, np.nan)
        if np.isnan(a) or np.isnan(p):
            continue
        target = symbol if p > a else safe
        if target not in w.columns:
            continue
        w.loc[d] = 0.0
        w.at[d, target] = 1.0

    return _settle(w)


def w_dual_momentum(prices: pd.DataFrame, risky: Optional[List[str]] = None,
                    safe: str = BONDS, lookback: int = TRADING_DAYS) -> pd.DataFrame:
    """Pick the stronger stock market, but only if it is beating bonds."""
    risky = risky or [BENCHMARK, INTERNATIONAL]
    risky = [s for s in risky if s in prices.columns]
    w = _blank(prices)
    if not risky or safe not in prices.columns:
        return w.fillna(0.0)

    trailing = prices / prices.shift(lookback) - 1.0

    for d in month_ends(prices.index):
        if d not in trailing.index:
            continue
        row = trailing.loc[d]
        scores = {s: row.get(s, np.nan) for s in risky}
        scores = {s: v for s, v in scores.items() if not np.isnan(v)}
        if not scores:
            continue
        best = max(scores, key=scores.get)
        safe_score = row.get(safe, np.nan)
        # Absolute momentum: the winner has to beat the safe asset too.
        target = best if (not np.isnan(safe_score) and scores[best] > safe_score) else safe
        w.loc[d] = 0.0
        w.at[d, target] = 1.0

    return _settle(w)


def w_xs_momentum(prices: pd.DataFrame, universe: List[str], top_n: int = 5,
                  lookback: int = 126, skip: int = 21,
                  safe: str = CASH) -> pd.DataFrame:
    """Hold the strongest names, skipping the most recent month.

    The skip is not decoration. Stocks tend to reverse over the very short
    term, so momentum measured right up to today is polluted by that
    reversal. Skipping the last month is standard in the literature.

    Anything with a negative trailing return goes to the safe asset instead,
    so the strategy is not forced to hold the least-bad losers in a bear
    market.
    """
    cols = [s for s in universe if s in prices.columns]
    w = _blank(prices)
    if not cols:
        return w.fillna(0.0)

    scores = prices.shift(skip) / prices.shift(skip + lookback) - 1.0

    for d in month_ends(prices.index):
        if d not in scores.index:
            continue
        row = scores.loc[d, cols].dropna()
        if row.empty:
            continue
        picks = row.sort_values(ascending=False).head(top_n)
        winners = [s for s in picks.index if picks[s] > 0]

        w.loc[d] = 0.0
        for s in winners:
            w.at[d, s] = 1.0 / top_n
        leftover = 1.0 - len(winners) / top_n
        if leftover > 1e-9 and safe in w.columns:
            w.at[d, safe] = leftover

    return _settle(w)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    label: str
    sim: pd.DataFrame


def run_all(bars: Dict[str, pd.DataFrame], watchlist: List[str],
            slippage_bps: float = 5.0, start_equity: float = 10_000.0,
            pullback_equity: Optional[pd.Series] = None) -> List[Entry]:
    prices = close_matrix(bars)

    # Every strategy has to be measured over the same window as the benchmark,
    # or the table compares a 20-year record against an 8-year one and calls
    # it a ranking. If some symbols have deeper history than SPY does, the
    # extra years are dropped rather than silently given to one strategy.
    if BENCHMARK in prices.columns:
        have = prices[BENCHMARK].dropna()
        if len(have):
            prices = prices[prices.index >= have.index[0]]

    honest = [s for s in HONEST_UNIVERSE if s in prices.columns]

    specs = [
        ("Buy and hold SPY", w_buy_hold(prices)),
        ("200-day trend filter", w_trend_filter(prices)),
        ("Dual momentum", w_dual_momentum(prices)),
    ]
    if len(honest) >= 6:
        specs.append(("Momentum, sector ETFs (clean)",
                      w_xs_momentum(prices, honest, top_n=3)))
    specs += [
        ("Momentum, top 5 (biased)", w_xs_momentum(prices, watchlist, top_n=5)),
        ("Equal weight watchlist (biased)", w_equal_weight(prices, watchlist)),
    ]

    out = []
    for label, w in specs:
        if float(w.abs().to_numpy().sum()) == 0.0:
            continue
        out.append(Entry(label, simulate(prices, w, slippage_bps, start_equity)))

    if pullback_equity is not None and len(pullback_equity) > 1:
        eq = pullback_equity.sort_index()
        sim = pd.DataFrame({
            "equity": eq,
            "ret": eq.pct_change().fillna(0.0),
            "exposure": 0.0,
            "turnover": 0.0,
        })
        out.append(Entry("Your trend pullback rules", sim))

    return out


def table(entries: List[Entry], start=None, end=None) -> pd.DataFrame:
    rows = []
    for e in entries:
        sim = e.sim
        if start is not None:
            sim = sim[sim.index >= pd.Timestamp(start)]
        if end is not None:
            sim = sim[sim.index <= pd.Timestamp(end)]
        if len(sim) < 2:
            continue
        sim = sim.copy()
        sim["equity"] = sim["equity"] / sim["equity"].iloc[0] * 10_000.0
        rows.append(metrics(sim, e.label))
    df = pd.DataFrame(rows)
    return df.sort_values("cagr_pct", ascending=False) if len(df) else df


SERIES = [
    ("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"),
    ("#eda100", "#c98500"), ("#e87ba4", "#d55181"), ("#008300", "#008300"),
]

_W, _H = 920, 340
_PL, _PR, _PT, _PB = 66, 12, 14, 30


def _multi_chart(entries: List[Entry], start=None, end=None):
    """One log-scale line per strategy, rebased to $10,000 at the start.

    Log scale, not linear. On a linear axis a strategy that ends higher
    visually crushes everything else and you cannot see what happened in the
    early years. On a log axis, equal vertical distance means equal
    percentage change, which is what you actually want to compare.
    """
    series = []
    for e in entries:
        sim = e.sim
        if start is not None:
            sim = sim[sim.index >= pd.Timestamp(start)]
        if end is not None:
            sim = sim[sim.index <= pd.Timestamp(end)]
        if len(sim) < 2:
            continue
        eq = sim["equity"] / sim["equity"].iloc[0] * 10_000.0
        series.append((e.label, eq))

    if not series:
        return "<p>Nothing to plot.</p>", "[]", ""

    common = series[0][1].index
    for _, eq in series[1:]:
        common = common.intersection(eq.index)
    common = common.sort_values()
    if len(common) < 2:
        return "<p>Nothing to plot.</p>", "[]", ""

    # Thin to at most ~400 points so the page stays small.
    step = max(1, len(common) // 400)
    idx = common[::step]

    vals = [(label, eq.reindex(idx).to_numpy(dtype=float)) for label, eq in series]
    allv = np.concatenate([v for _, v in vals])
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    lo, hi = max(lo * 0.92, 1.0), hi * 1.08
    llo, lhi = np.log10(lo), np.log10(hi)

    def ypix(v):
        return _H - _PB - (np.log10(np.clip(v, 1.0, None)) - llo) / (lhi - llo) * (_H - _PB - _PT)

    xs = np.linspace(_PL, _W - _PR, len(idx))

    grid, ticks = [], []
    for k in range(5):
        lv = llo + (lhi - llo) * k / 4
        v = 10 ** lv
        y = ypix(v)
        grid.append(f'<line x1="{_PL}" y1="{y:.1f}" x2="{_W-_PR}" y2="{y:.1f}" '
                    f'stroke="var(--gridline)" stroke-width="1"/>')
        ticks.append(f'<text x="{_PL-8}" y="{y+4:.1f}" text-anchor="end" '
                     f'class="tick">${v:,.0f}</text>')

    xlab = []
    for k, anchor in ((0, "start"), (len(idx)//2, "middle"), (len(idx)-1, "end")):
        xlab.append(f'<text x="{xs[k]:.1f}" y="{_H-8}" text-anchor="{anchor}" '
                    f'class="tick">{idx[k].strftime("%Y")}</text>')

    paths, legend, payload = [], [], []
    for i, (label, v) in enumerate(vals):
        ys = ypix(v)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys) if not np.isnan(y))
        paths.append(f'<polyline points="{pts}" fill="none" stroke="var(--s{i})" '
                     f'stroke-width="2" stroke-linejoin="round" '
                     f'vector-effect="non-scaling-stroke"/>')
        legend.append(f'<span class="lg"><i style="background:var(--s{i})"></i>'
                      f'{label}</span>')
        payload.append({"label": label, "v": [round(float(x), 0) for x in v]})

    svg = f"""<svg viewBox="0 0 {_W} {_H}" class="chart" preserveAspectRatio="none">
  {''.join(grid)}{''.join(paths)}{''.join(ticks)}{''.join(xlab)}
  <line class="crosshair" x1="0" y1="{_PT}" x2="0" y2="{_H-_PB}"
        stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3" opacity="0"/>
</svg>"""

    import json as _json
    data = _json.dumps({
        "x": [round(float(x), 1) for x in xs],
        "d": [d.strftime("%b %Y") for d in idx],
        "s": payload,
    })
    return svg, data, "".join(legend)


def _table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No results.</p>"
    head = ("<tr><th>Strategy</th><th class='n'>CAGR</th><th class='n'>Max drawdown</th>"
            "<th class='n'>Sharpe</th><th class='n'>Calmar</th>"
            "<th class='n'>Worst year</th><th class='n'>In market</th>"
            "<th class='n'>$10k became</th></tr>")
    rows = ""
    for _, r in df.iterrows():
        rows += (f"<tr><td>{r['strategy']}</td>"
                 f"<td class='n'>{r['cagr_pct']:.1f}%</td>"
                 f"<td class='n neg'>-{r['max_dd_pct']:.1f}%</td>"
                 f"<td class='n'>{r['sharpe']:.2f}</td>"
                 f"<td class='n'>{r['calmar']:.2f}</td>"
                 f"<td class='n {'neg' if r['worst_year_pct'] < 0 else ''}'>"
                 f"{r['worst_year_pct']:.1f}%</td>"
                 f"<td class='n'>{r['exposure_pct']:.0f}%</td>"
                 f"<td class='n'>${r['final']:,.0f}</td></tr>")
    return f"<div class='card scroll'><table><thead>{head}</thead><tbody>{rows}</tbody></table></div>"


def compare_html(entries: List[Entry], full: pd.DataFrame, ins: pd.DataFrame,
                 oos: pd.DataFrame, split: str) -> str:
    svg, data, legend = _multi_chart(entries)
    vars_light = "".join(f"--s{i}:{c[0]};" for i, c in enumerate(SERIES))
    vars_dark = "".join(f"--s{i}:{c[1]};" for i, c in enumerate(SERIES))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strategy comparison</title>
<style>
 :root {{ color-scheme:light; --surface-1:#fcfcfb; --plane:#f9f9f7;
   --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
   --gridline:#e1e0d9; --border:rgba(11,11,11,0.10); --critical:#d03b3b;
   --good:#006300; {vars_light} }}
 @media (prefers-color-scheme:dark) {{ :root:not([data-theme="light"]) {{
   color-scheme:dark; --surface-1:#1a1a19; --plane:#0d0d0d;
   --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
   --gridline:#2c2c2a; --border:rgba(255,255,255,0.10); --critical:#e06060;
   --good:#0ca30c; {vars_dark} }} }}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--plane);color:var(--text-primary);
   font:14.5px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
 .wrap{{max-width:1000px;margin:0 auto;padding:28px 18px 60px}}
 h1{{font-size:22px;margin:0 0 6px;letter-spacing:-0.01em}}
 h2{{font-size:15px;margin:30px 0 10px}}
 .sub{{color:var(--text-secondary);margin-bottom:20px}}
 .card{{background:var(--surface-1);border:1px solid var(--border);
   border-radius:10px;padding:14px;position:relative}}
 .chart{{width:100%;height:340px;display:block}}
 .tick{{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}}
 .legend{{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:12px;
   font-size:12.5px;color:var(--text-secondary)}}
 .lg{{display:inline-flex;align-items:center;gap:6px}}
 .lg i{{width:11px;height:3px;border-radius:2px;display:inline-block}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 th{{text-align:left;color:var(--text-secondary);font-weight:500;
   border-bottom:1px solid var(--border);padding:8px;white-space:nowrap}}
 td{{padding:8px;border-bottom:1px solid var(--gridline)}}
 tr:last-child td{{border-bottom:none}}
 .n{{text-align:right;font-variant-numeric:tabular-nums}}
 .neg{{color:var(--critical)}}
 .tip{{position:absolute;pointer-events:none;opacity:0;background:var(--surface-1);
   border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12px;
   box-shadow:0 4px 16px rgba(0,0,0,.16);white-space:nowrap;z-index:4;
   font-variant-numeric:tabular-nums}}
 .tip b{{display:block;margin-bottom:4px;color:var(--text-secondary);font-weight:500}}
 .tip i{{width:9px;height:3px;border-radius:2px;display:inline-block;margin-right:6px}}
 .note{{background:var(--surface-1);border:1px solid var(--border);
   border-left:3px solid var(--critical);border-radius:9px;padding:14px 16px;
   margin-top:26px;color:var(--text-secondary);font-size:13.5px}}
 .scroll{{overflow-x:auto}}
</style></head><body><div class="wrap">
<h1>Strategy comparison</h1>
<div class="sub">Same data, same costs, same starting balance. Growth of $10,000,
on a log scale so equal vertical distance means equal percentage change.</div>

<div class="card" id="c">{svg}<div class="tip" id="t"></div></div>
<div class="legend">{legend}</div>

<h2>Full period</h2>
{_table_html(full)}

<h2>Out of sample, {split} onward</h2>
<div class="sub" style="margin:-4px 0 10px;font-size:13px">This is the column
that matters. Every strategy here was published before this date, so this
stretch is data none of them were designed against.</div>
{_table_html(oos)}

<h2>Earlier period, up to {split}</h2>
{_table_html(ins)}

<div class="note">
<strong>How to read this without fooling yourself.</strong>
The strategy at the top is partly the luckiest, not purely the best. Running
six strategies means the winner beat five others by chance as well as by
merit. A strategy that leads the full period but not the out-of-sample period
is telling you something, and what it is telling you is bad.
<br><br>
Max drawdown matters more than CAGR. A strategy you abandon at the bottom
returns whatever you earned before you quit, which is usually a loss. Ask
yourself honestly whether you would still be running the top strategy after
the worst year listed in its row.
<br><br>
None of this is a prediction. These are historical results under fixed cost
assumptions, and your real fills will be worse than the model. Past
performance does not tell you what happens next.
</div>
</div>
<script>
(function(){{
 var D = {data};
 var card=document.getElementById('c'), tip=document.getElementById('t');
 if(!card||!D.x.length) return;
 var svg=card.querySelector('svg'), cross=svg.querySelector('.crosshair');
 function show(cx){{
  var r=svg.getBoundingClientRect(), vx=(cx-r.left)/r.width*{_W};
  var b=0,bd=Infinity;
  for(var i=0;i<D.x.length;i++){{var d=Math.abs(D.x[i]-vx); if(d<bd){{bd=d;b=i;}}}}
  cross.setAttribute('x1',D.x[b]); cross.setAttribute('x2',D.x[b]);
  cross.setAttribute('opacity','1');
  var rows=D.s.map(function(s,j){{
    return '<i style="background:var(--s'+j+')"></i>'+s.label+' &nbsp;<b style="display:inline;color:inherit">$'
      + (s.v[b]||0).toLocaleString() + '</b>';
  }}).join('<br>');
  tip.innerHTML='<b>'+D.d[b]+'</b>'+rows;
  tip.style.opacity='1';
  var px=D.x[b]/{_W}*r.width;
  tip.style.left=Math.min(Math.max(px+12,4), Math.max(r.width-230,4))+'px';
  tip.style.top='14px';
 }}
 card.addEventListener('mousemove',function(e){{show(e.clientX);}});
 card.addEventListener('touchstart',function(e){{show(e.touches[0].clientX);}},{{passive:true}});
 card.addEventListener('touchmove',function(e){{show(e.touches[0].clientX);}},{{passive:true}});
 function hide(){{tip.style.opacity='0';cross.setAttribute('opacity','0');}}
 card.addEventListener('mouseleave',hide);
 card.addEventListener('touchend',hide);
}})();
</script></body></html>"""


def print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print("-" * 92)
    if df.empty:
        print("  no results")
        return
    print(f"  {'Strategy':<26}{'CAGR':>8}{'Max DD':>9}{'Sharpe':>8}"
          f"{'Calmar':>8}{'Worst yr':>10}{'In mkt':>8}{'$10k ->':>13}")
    for _, r in df.iterrows():
        print(f"  {r['strategy']:<26}{r['cagr_pct']:>7.1f}%{-r['max_dd_pct']:>8.1f}%"
              f"{r['sharpe']:>8.2f}{r['calmar']:>8.2f}{r['worst_year_pct']:>9.1f}%"
              f"{r['exposure_pct']:>7.0f}%{r['final']:>12,.0f}")
