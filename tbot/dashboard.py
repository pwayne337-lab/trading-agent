"""
The scoreboard.

Builds a single self-contained HTML page from the agent's recorded state.
No server, no database, no JavaScript framework. Everything the page shows
came from a real run, so the page cannot report a trade that did not happen.

The most important thing on it is the freshness banner. A dashboard that
quietly shows three-day-old numbers while you believe they are live is worse
than no dashboard at all.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .report import _line_chart
from .state import load_equity_history, load_state

SITE = Path(__file__).resolve().parent.parent / "site"
SITE.mkdir(parents=True, exist_ok=True)


def _age(updated_at) -> tuple:
    """Returns (human string, hours, severity)."""
    if not updated_at:
        return "never run", 1e9, "critical"
    try:
        t = datetime.fromisoformat(str(updated_at))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
    except Exception:
        return "unknown", 1e9, "critical"

    hrs = (datetime.now(timezone.utc) - t).total_seconds() / 3600
    # A run stamped slightly ahead of this clock is clock skew, not a run from
    # the future. Never render it as negative minutes.
    if hrs < 0:
        hrs = 0.0
    if hrs < 1:
        s = f"{int(hrs * 60)} min ago"
    elif hrs < 48:
        s = f"{int(hrs)} hours ago"
    else:
        s = f"{int(hrs / 24)} days ago"

    sev = "good" if hrs < 30 else ("warning" if hrs < 96 else "critical")
    return s, hrs, sev


def _tile(label, value, sub="", tone=""):
    return (f'<div class="tile {tone}"><div class="tl">{label}</div>'
            f'<div class="tv">{value}</div><div class="ts">{sub}</div></div>')


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_html(state: dict = None, history: list = None,
               title: str = "Trading agent") -> str:
    state = state if state is not None else load_state()
    history = history if history is not None else load_equity_history()

    acct = state.get("account") or {}
    equity = float(acct.get("equity") or 0)
    cash = float(acct.get("cash") or 0)
    mode = str(state.get("mode", "unknown")).upper()
    positions = state.get("positions") or []
    signals = state.get("signals") or []
    vetoes = state.get("vetoes") or []
    orders = state.get("orders") or []
    skipped = state.get("skipped") or []
    errors = state.get("errors") or []

    age_str, age_hrs, age_sev = _age(state.get("updated_at"))

    # Day change from the equity history.
    day_change = 0.0
    if len(history) >= 2 and history[-2]["equity"] > 0:
        day_change = (history[-1]["equity"] / history[-2]["equity"] - 1) * 100
    total_change = 0.0
    if history and history[0]["equity"] > 0:
        total_change = (equity / history[0]["equity"] - 1) * 100

    open_pl = sum(float(p.get("unrealized_pl") or 0) for p in positions)

    # -- banner -------------------------------------------------------------
    if not state.get("updated_at"):
        banner = ('<div class="banner warning"><strong>Nothing here yet.</strong> '
                  'The agent has not run. This page fills in the first time it does.</div>')
    elif age_sev == "critical":
        banner = (f'<div class="banner critical"><strong>Stale.</strong> '
                  f'The agent last ran {age_str}. Nothing on this page is current. '
                  f'Check that the scheduled run is still working.</div>')
    elif age_sev == "warning":
        banner = (f'<div class="banner warning"><strong>Getting old.</strong> '
                  f'Last run {age_str}. Fine over a weekend or a market holiday, '
                  f'not fine on a Wednesday.</div>')
    else:
        banner = ""

    # The watchers come first. A missing stop order matters more than the
    # equity number sitting under it.
    findings = state.get("findings") or []
    for f in findings:
        sev = f.get("severity")
        if sev not in ("critical", "warning"):
            continue
        cls = "critical" if sev == "critical" else "warning"
        label = "Needs attention." if sev == "critical" else "Worth a look."
        banner += (f'<div class="banner {cls}"><strong>{label}</strong> '
                   f'{_esc(f.get("message"))}</div>')

    if errors and not findings:
        banner += ('<div class="banner critical"><strong>Errors on the last run.</strong> '
                   + "; ".join(_esc(e) for e in errors[:3]) + '</div>')

    if mode == "LIVE":
        banner = ('<div class="banner critical"><strong>LIVE MONEY.</strong> '
                  'This agent is trading a funded account.</div>') + banner

    # -- charts -------------------------------------------------------------
    if len(history) >= 2:
        dates = [h["date"][5:] for h in history]
        vals = [h["equity"] for h in history]
        chart, pts = _line_chart(dates, vals, "--series-1")
        chart_block = f'<div class="card" id="c1">{chart}<div class="tip" id="t1"></div></div>'
    else:
        chart, pts = "", "[]"
        chart_block = ('<div class="card empty">Not enough history to plot yet. '
                       'The chart appears after the agent has run on two separate days.</div>')

    # -- tiles --------------------------------------------------------------
    tiles = "".join([
        _tile("Account equity", f"${equity:,.2f}",
              f"{mode.lower()} account &middot; {len(history)} run(s) recorded"),
        _tile("Today", f"{day_change:+.2f}%", "since the previous run",
              "good" if day_change > 0 else ("bad" if day_change < 0 else "")),
        _tile("Since start", f"{total_change:+.2f}%", "of tracking",
              "good" if total_change > 0 else ("bad" if total_change < 0 else "")),
        _tile("Open positions", f"{len(positions)}",
              f"open P&amp;L ${open_pl:+,.2f}",
              "good" if open_pl > 0 else ("bad" if open_pl < 0 else "")),
        _tile("Cash", f"${cash:,.2f}", "uninvested"),
        _tile("Last run", age_str,
              f"{len(orders)} order(s), {len(vetoes)} blocked", age_sev),
    ])

    # -- positions ----------------------------------------------------------
    if positions:
        rows = "".join(
            f"<tr><td><strong>{_esc(p['symbol'])}</strong></td>"
            f"<td class='n'>{p.get('shares', 0)}</td>"
            f"<td class='n'>${float(p.get('avg_entry') or 0):,.2f}</td>"
            f"<td class='n'>${float(p.get('market_value') or 0):,.2f}</td>"
            f"<td class='n {'pos' if float(p.get('unrealized_pl') or 0) >= 0 else 'neg'}'>"
            f"${float(p.get('unrealized_pl') or 0):+,.2f}</td></tr>"
            for p in positions)
        pos_block = (f'<div class="card scroll"><table><thead><tr><th>Symbol</th>'
                     f'<th class="n">Shares</th><th class="n">Avg cost</th>'
                     f'<th class="n">Value</th><th class="n">Open P&amp;L</th>'
                     f'</tr></thead><tbody>{rows}</tbody></table></div>')
    else:
        pos_block = '<div class="card empty">Holding nothing right now.</div>'

    # -- today --------------------------------------------------------------
    items = []
    for e in (state.get("exits") or []):
        pl = e.get("unrealized_pl")
        pl_txt = f", P&amp;L ${float(pl):+,.2f}" if pl is not None else ""
        items.append(
            f'<li class="ev sold"><span class="tag">sold</span> '
            f'<strong>{_esc(e.get("symbol"))}</strong> {_esc(e.get("reason"))}'
            f'{pl_txt}</li>')
    for o in orders:
        items.append(
            f'<li class="ev ok"><span class="tag">bought</span> '
            f'<strong>{_esc(o.get("symbol"))}</strong> {o.get("shares", 0)} sh, '
            f'stop ${float(o.get("stop") or 0):,.2f}, '
            f'target ${float(o.get("target") or 0):,.2f}, '
            f'risking ${float(o.get("dollars_at_risk") or 0):,.2f}</li>')
    for v in vetoes:
        items.append(
            f'<li class="ev block"><span class="tag">blocked</span> '
            f'<strong>{_esc(v.get("symbol"))}</strong> {_esc(v.get("reason"))}</li>')
    for s in skipped:
        items.append(
            f'<li class="ev skip"><span class="tag">skipped</span> '
            f'<strong>{_esc(s.get("symbol"))}</strong> {_esc(s.get("reason"))}</li>')
    if not items:
        items.append('<li class="ev quiet">No setups met the rules. '
                     'Most days look like this.</li>')
    today_block = f'<div class="card"><ul class="events">{"".join(items)}</ul></div>'

    # -- briefing -----------------------------------------------------------
    brief = state.get("briefing") or ""
    if brief:
        paras = "".join(f"<p>{_esc(p)}</p>" for p in brief.split("\n") if p.strip())
        brief_block = f'<div class="card brief">{paras}</div>'
    else:
        brief_block = ('<div class="card empty">No briefing. Add an Anthropic API '
                       'key to .env to turn this on.</div>')

    # -- recent trades ------------------------------------------------------
    rt = state.get("recent_trades") or []
    if rt:
        rows = "".join(
            f"<tr><td><strong>{_esc(t.get('symbol'))}</strong></td>"
            f"<td>{_esc(t.get('closed'))}</td>"
            f"<td>{_esc(t.get('reason'))}</td>"
            f"<td class='n {'pos' if float(t.get('pnl') or 0) >= 0 else 'neg'}'>"
            f"${float(t.get('pnl') or 0):+,.2f}</td></tr>" for t in rt[:20])
        trades_block = (f'<div class="card scroll"><table><thead><tr><th>Symbol</th>'
                        f'<th>Closed</th><th>Why</th><th class="n">P&amp;L</th>'
                        f'</tr></thead><tbody>{rows}</tbody></table></div>')
    else:
        trades_block = '<div class="card empty">No closed trades yet.</div>'

    stamp = (f"Generated {state['updated_at']} UTC from the agent's own run records"
             if state.get("updated_at") else "This page has not been generated from a real run yet")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>{_esc(title)}</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1:#fcfcfb; --plane:#f9f9f7;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
    --gridline:#e1e0d9; --border:rgba(11,11,11,0.10);
    --series-1:#2a78d6; --good:#006300; --critical:#d03b3b; --warning:#a86a00;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface-1:#1a1a19; --plane:#0d0d0d;
      --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
      --gridline:#2c2c2a; --border:rgba(255,255,255,0.10);
      --series-1:#3987e5; --good:#0ca30c; --critical:#e06060; --warning:#fab219;
    }}
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--plane);color:var(--text-primary);
    font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
    -webkit-text-size-adjust:100%}}
  .wrap{{max-width:900px;margin:0 auto;padding:20px 16px 56px}}
  header{{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline;
    justify-content:space-between;margin-bottom:16px}}
  h1{{font-size:20px;margin:0;letter-spacing:-0.01em}}
  .mode{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;
    border:1px solid var(--border);border-radius:999px;padding:3px 10px;
    color:var(--text-secondary)}}
  .mode.live{{color:var(--critical);border-color:var(--critical)}}
  h2{{font-size:14px;margin:26px 0 9px;color:var(--text-secondary);font-weight:600}}
  .banner{{border-radius:9px;padding:11px 14px;margin-bottom:14px;font-size:13.5px;
    border:1px solid var(--border);background:var(--surface-1)}}
  .banner.critical{{border-left:3px solid var(--critical)}}
  .banner.warning{{border-left:3px solid var(--warning)}}
  .tiles{{display:grid;gap:9px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}}
  .tile{{background:var(--surface-1);border:1px solid var(--border);
    border-radius:10px;padding:12px 14px}}
  .tl{{color:var(--text-secondary);font-size:11.5px}}
  .tv{{font-size:22px;font-weight:600;margin:3px 0 2px;letter-spacing:-0.02em}}
  .ts{{color:var(--muted);font-size:11px}}
  .tile.good .tv{{color:var(--good)}} .tile.bad .tv,.tile.critical .tv{{color:var(--critical)}}
  .tile.warning .tv{{color:var(--warning)}}
  .card{{background:var(--surface-1);border:1px solid var(--border);
    border-radius:10px;padding:14px;position:relative}}
  .card.empty{{color:var(--muted);font-size:13.5px}}
  .chart{{width:100%;height:230px;display:block}}
  .tick{{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}}
  .tip{{position:absolute;pointer-events:none;opacity:0;background:var(--surface-1);
    border:1px solid var(--border);border-radius:8px;padding:6px 9px;font-size:12px;
    box-shadow:0 4px 14px rgba(0,0,0,.14);transition:opacity .1s;white-space:nowrap}}
  table{{width:100%;border-collapse:collapse;font-size:13.5px}}
  th{{text-align:left;color:var(--text-secondary);font-weight:500;
    border-bottom:1px solid var(--border);padding:7px 8px;white-space:nowrap}}
  td{{padding:8px;border-bottom:1px solid var(--gridline)}}
  tr:last-child td{{border-bottom:none}}
  .n{{text-align:right;font-variant-numeric:tabular-nums}}
  .pos{{color:var(--good)}} .neg{{color:var(--critical)}}
  .scroll{{overflow-x:auto}}
  .events{{list-style:none;margin:0;padding:0}}
  .ev{{padding:9px 0;border-bottom:1px solid var(--gridline);font-size:13.5px}}
  .ev:last-child{{border-bottom:none}}
  .ev.quiet{{color:var(--muted)}}
  .tag{{display:inline-block;font-size:10.5px;text-transform:uppercase;
    letter-spacing:.06em;padding:2px 7px;border-radius:999px;margin-right:7px;
    border:1px solid var(--border);color:var(--text-secondary)}}
  .ev.ok .tag{{color:var(--good);border-color:var(--good)}}
  .ev.block .tag{{color:var(--critical);border-color:var(--critical)}}
  .ev.sold .tag{{color:var(--warning);border-color:var(--warning)}}
  .brief p{{margin:0 0 10px}} .brief p:last-child{{margin:0}}
  footer{{margin-top:30px;color:var(--muted);font-size:11.5px;line-height:1.6}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{_esc(title)}</h1>
    <span class="mode {'live' if mode == 'LIVE' else ''}">{_esc(mode)}</span>
  </header>

  {banner}

  <div class="tiles">{tiles}</div>

  <h2>Account equity</h2>
  {chart_block}

  <h2>Open positions</h2>
  {pos_block}

  <h2>What the agent did on its last run</h2>
  {today_block}

  <h2>Briefing</h2>
  {brief_block}

  <h2>Recently closed</h2>
  {trades_block}

  <footer>
    {_esc(stamp)}.
    Numbers update only when the agent runs; this page does not poll anything.<br>
    A paper account is fake money. Results here are not a prediction of what
    real money would do, and past results do not tell you what comes next.
  </footer>
</div>
<script>
(function(){{
  var pts = {pts};
  var card = document.getElementById('c1'), tip = document.getElementById('t1');
  if (!card || !pts.length) return;
  var svg = card.querySelector('svg');
  var cross = svg.querySelector('.crosshair'), dot = svg.querySelector('.cursor-dot');
  function at(clientX) {{
    var r = svg.getBoundingClientRect();
    var vx = (clientX - r.left) / r.width * 920;
    var best = 0, bd = Infinity;
    for (var i = 0; i < pts.length; i++) {{
      var d = Math.abs(pts[i].x - vx);
      if (d < bd) {{ bd = d; best = i; }}
    }}
    var p = pts[best];
    cross.setAttribute('x1', p.x); cross.setAttribute('x2', p.x);
    cross.setAttribute('opacity', '1');
    dot.setAttribute('cx', p.x); dot.setAttribute('cy', p.y);
    dot.setAttribute('opacity', '1');
    tip.innerHTML = p.d + ' &middot; <b>$' +
      p.v.toLocaleString(undefined, {{maximumFractionDigits: 0}}) + '</b>';
    tip.style.opacity = '1';
    tip.style.left = Math.min(Math.max(p.x / 920 * r.width - 48, 4), r.width - 140) + 'px';
    tip.style.top = Math.max(p.y / 280 * r.height - 40, 2) + 'px';
  }}
  card.addEventListener('mousemove', function(e){{ at(e.clientX); }});
  card.addEventListener('touchstart', function(e){{ at(e.touches[0].clientX); }}, {{passive:true}});
  card.addEventListener('touchmove', function(e){{ at(e.touches[0].clientX); }}, {{passive:true}});
  function hide(){{ tip.style.opacity='0'; cross.setAttribute('opacity','0');
    dot.setAttribute('opacity','0'); }}
  card.addEventListener('mouseleave', hide);
  card.addEventListener('touchend', hide);
}})();
</script>
</body>
</html>"""


def write_dashboard(title: str = "Trading agent") -> Path:
    path = SITE / "index.html"
    path.write_text(build_html(title=title))
    return path
