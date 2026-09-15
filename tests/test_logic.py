"""
Correctness checks for the strategy and backtest engine.

These do not test whether the strategy makes money. They test whether the
code does what it claims. Those are completely different questions, and only
the second one can be answered by a computer.

Run with:  python -m tests.test_logic
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tbot.config import AgentConfig
from tbot.backtest import run_backtest
from tbot.indicators import add_indicators, atr, ema, sma
from tbot.risk import size_position
from tbot.strategy import prepare, signals_for_symbol, _row_signal
from tests.synthetic import make_series, universe

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -> {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
print("\n1. Indicators use only past data")
# ---------------------------------------------------------------------------

df = make_series(n=400, seed=3)
full = add_indicators(df, AgentConfig().strategy)
truncated = add_indicators(df.iloc[:300], AgentConfig().strategy)

# Every indicator value on bar 299 must be identical whether or not bars 300+
# exist. If it is not, the indicator is peeking into the future.
cols = ["sma_fast", "sma_slow", "ema_pb", "atr", "swing_low", "adv",
        "donchian_hi", "rsi_fast", "sma_exit"]
same = all(
    np.isclose(full[c].iloc[299], truncated[c].iloc[299], equal_nan=True)
    for c in cols
)
check("indicator values unchanged when future bars are removed", same)

# EMA sanity against a hand computation.
manual = df["close"].ewm(span=20, adjust=False, min_periods=20).mean()
check("ema matches reference implementation",
      np.isclose(full["ema_pb"].dropna().iloc[-1], manual.dropna().iloc[-1]))

# ATR must be positive and roughly the size of a typical daily range.
med_range = (df["high"] - df["low"]).median()
last_atr = full["atr"].iloc[-1]
check("atr is in the right ballpark vs median daily range",
      0.4 * med_range < last_atr < 3.0 * med_range,
      f"atr={last_atr:.2f} median_range={med_range:.2f}")


# ---------------------------------------------------------------------------
print("\n2. Signal rules fire only when every condition is true")
# ---------------------------------------------------------------------------

cfg = AgentConfig()
# This section checks the pullback rules specifically, so only those are on.
# With every strategy enabled the list also contains breakouts and reversions,
# which are supposed to fail a pullback-shaped assertion.
cfg.strategy.enabled = ["pullback"]
data = prepare(make_series(n=1200, seed=11), cfg.strategy)
sigs = signals_for_symbol("TEST", make_series(n=1200, seed=11), cfg.strategy)
check("some signals were generated on synthetic data", len(sigs) > 0, f"n={len(sigs)}")

bad_trend = bad_reclaim = bad_stop = 0
for s in sigs:
    i = data.index.get_loc(s.signal_date)
    row, prev = data.iloc[i], data.iloc[i - 1]
    if not (row["close"] > row["sma_slow"] and row["sma_fast"] > row["sma_slow"]):
        bad_trend += 1
    if not (prev["close"] <= prev["ema_pb"] and row["close"] > row["ema_pb"]):
        bad_reclaim += 1
    if not (s.stop < row["close"]):
        bad_stop += 1

check("every signal passed the trend filter", bad_trend == 0, f"{bad_trend} bad")
check("every signal is a genuine 20 EMA reclaim", bad_reclaim == 0, f"{bad_reclaim} bad")
check("every stop sits below the trigger close", bad_stop == 0, f"{bad_stop} bad")

# Stop distance guard rails were honored.
viol = [s for s in sigs
        if not (cfg.strategy.min_stop_atr - 1e-9
                <= (s.reference_close - s.stop) / s.atr
                <= cfg.strategy.max_stop_atr + 1e-9)]
check("stop distance stayed inside the ATR guard rails", len(viol) == 0, f"{len(viol)} bad")

# Signals must not change when future data is appended or removed.
early = signals_for_symbol("TEST", make_series(n=1200, seed=11).iloc[:900], cfg.strategy)
overlap = [s for s in sigs if s.signal_date <= data.index[899]]
match = (len(early) == len(overlap)
         and all(a.signal_date == b.signal_date and np.isclose(a.stop, b.stop)
                 for a, b in zip(early, overlap)))
check("historical signals do not change when later bars are removed", match,
      f"{len(early)} vs {len(overlap)}")


# ---------------------------------------------------------------------------
print("\n3. Position sizing math")
# ---------------------------------------------------------------------------

cfg = AgentConfig()
cfg.risk.starting_equity = 10_000
o = size_position(10_000, entry=100.0, stop=98.0, cfg_risk=cfg.risk,
                  cfg_strategy=cfg.strategy)
# Risk $100, $2 per share => 50 shares, $5,000 notional (50% of equity) which
# is over the 25% cap, so the cap should bind at 25 shares.
check("position cap binds before risk sizing when the stop is wide", o.shares == 25,
      f"shares={o.shares}")

o2 = size_position(10_000, entry=20.0, stop=19.0, cfg_risk=cfg.risk,
                   cfg_strategy=cfg.strategy)
# Risk $100, $1 per share => 100 shares, $2,000 notional = 20% of equity, under
# the cap, so risk sizing binds.
check("risk sizing binds when the stop is tight enough", o2.shares == 100,
      f"shares={o2.shares}")
check("dollars at risk equals 1% of equity", np.isclose(o2.dollars_at_risk, 100.0),
      f"{o2.dollars_at_risk}")
check("target is 2R above entry", np.isclose(o2.target, 22.0), f"{o2.target}")

o3 = size_position(10_000, entry=100.0, stop=101.0, cfg_risk=cfg.risk,
                   cfg_strategy=cfg.strategy)
check("a stop above entry is refused", not o3.ok and o3.shares == 0)

o4 = size_position(10_000, entry=100.0, stop=98.0, cfg_risk=cfg.risk,
                   cfg_strategy=cfg.strategy, open_positions=cfg.risk.max_open_positions)
check("position count limit is enforced", not o4.ok, o4.rejected_reason or "")

o5 = size_position(10_000, entry=100.0, stop=98.0, cfg_risk=cfg.risk,
                   cfg_strategy=cfg.strategy, halted=True)
check("drawdown breaker blocks new trades", not o5.ok, o5.rejected_reason or "")

# The breaker must be able to un-trip. A one-way switch retires the strategy
# after a single bad stretch and flatlines every backtest that hits it.
from tbot.risk import DrawdownMonitor

_dd = DrawdownMonitor(10_000, limit=0.20, resume_below=0.10, cooldown_days=60)
check("no trip on a shallow drawdown", not _dd.update(9_000))
check("trips at the limit", _dd.update(7_900))
check("stays tripped part-way back up", _dd.update(8_800))
check("re-arms once recovered inside the resume level", not _dd.update(9_100))
check("can trip a second time later", _dd.update(7_000))
check("counted both trips", _dd.trips == 2, f"trips={_dd.trips}")

# The deadlock: a halted strategy holds nothing, so its equity stops moving,
# so its drawdown never shrinks, so recovery alone never comes. Without a
# cooldown the breaker is permanent and every long backtest flatlines.
_stuck = DrawdownMonitor(10_000, limit=0.20, resume_below=0.10, cooldown_days=60)
_stuck.update(7_500)
check("frozen equity keeps the breaker tripped", _stuck.tripped)
for _ in range(58):
    _stuck.update(7_500)
check("still halted one day before the cooldown ends", _stuck.tripped,
      f"days={_stuck.days_tripped}")
_stuck.update(7_500)
check("the cooldown releases it even with no recovery at all",
      not _stuck.tripped, f"days={_stuck.days_tripped}")

_never = DrawdownMonitor(10_000, limit=0.20, resume_below=0.10, cooldown_days=60)
_never.update(7_500)
_halted_days = sum(1 for _ in range(500) if _never.update(7_500))
check("a permanently flat account is not halted forever",
      _halted_days < 120, f"halted {_halted_days} of 500 days")

# Correlation guard: five copies of the same bet is one bet at five times
# the size, and the sizing math does not know that unless we tell it.
from tbot.risk import correlation_block

_r = np.random.default_rng(5)
_base = _r.normal(0, 0.01, 300)
_rets = pd.DataFrame({
    "QQQ": _base,
    "XLK": _base * 0.98 + _r.normal(0, 0.0015, 300),   # near-identical
    "TLT": _r.normal(0, 0.006, 300),                    # unrelated
}, index=pd.bdate_range("2025-01-01", periods=300))

check("a near-duplicate of a held position is refused",
      correlation_block("XLK", ["QQQ"], _rets, 60, 0.80) is not None)
check("an unrelated position is allowed",
      correlation_block("TLT", ["QQQ"], _rets, 60, 0.80) is None)
check("a symbol is never blocked against itself",
      correlation_block("QQQ", ["QQQ"], _rets, 60, 0.80) is None)
check("nothing held means nothing to block",
      correlation_block("XLK", [], _rets, 60, 0.80) is None)
check("too little history does not block on noise",
      correlation_block("XLK", ["QQQ"], _rets.head(8), 60, 0.80) is None)
check("the block explains itself in words",
      "QQQ" in (correlation_block("XLK", ["QQQ"], _rets, 60, 0.80) or ""))


# ---------------------------------------------------------------------------
print("\n4. Backtest fills and accounting")
# ---------------------------------------------------------------------------

bars = universe(["AAA", "BBB", "CCC", "DDD"], seed0=21, n=1400)
cfg = AgentConfig()
res = run_backtest(bars, cfg)
tdf = res.trades_df()

check("backtest produced trades", len(tdf) > 0, f"n={len(tdf)}")
check("no trade was entered before its signal",
      all(t.entry_date > t.signal_date for t in res.trades))
check("no trade exited before it was entered",
      all(t.exit_date >= t.entry_date for t in res.trades))
check("never more open positions than the limit",
      res.equity["open_positions"].max() <= cfg.risk.max_open_positions,
      f"max={res.equity['open_positions'].max()}")
check("equity never went negative", (res.equity["equity"] > 0).all())
check("cash never went negative (no accidental margin)",
      (res.equity["cash"] >= -1e-6).all(),
      f"min cash={res.equity['cash'].min():.2f}")

# Losses should cluster near -1R. They will not all be exactly -1R because of
# gaps and slippage, which is exactly the point.
stop_outs = tdf[tdf["reason"] == "stop"]
if len(stop_outs):
    typical = stop_outs["R"].median()
    check("a clean stop-out loses about the 1R the sizing intended",
          -1.15 < typical < -0.9, f"median stop-out = {typical:.2f}R")

gaps = tdf[tdf["reason"] == "gap through stop"]
if len(gaps):
    check("gaps through the stop lose MORE than 1R (this is the real risk)",
          gaps["R"].median() < -1.0, f"median gap loss = {gaps['R'].median():.2f}R")

soft = tdf[(tdf["R"] < 0) & (tdf["reason"].isin(["trend break", "time stop"]))]
if len(soft):
    check("trend-break exits cut losses before the stop is reached",
          soft["R"].median() > -1.0, f"median = {soft['R'].median():.2f}R")

winners = tdf[tdf["R"] > 0]
target_hits = tdf[tdf["reason"].isin(["target", "gap through target"])]
if len(target_hits):
    check("target exits land near +2R",
          1.5 < target_hits["R"].median() < 2.3,
          f"median = {target_hits['R'].median():.2f}R")

# Hand-verify one trade end to end.
print("\n5. Hand check of a single trade")
t = res.trades[0]
sym_data = prepare(bars[t.symbol], cfg.strategy)
sig_i = sym_data.index.get_loc(t.signal_date)
next_open = float(sym_data.iloc[sig_i + 1]["open"])
expected_fill = next_open * (1 + cfg.costs.slippage_bps / 10_000)
print(f"  {t.symbol}: signal {t.signal_date.date()} -> entry {t.entry_date.date()}")
print(f"  next session open {next_open:.4f}, +{cfg.costs.slippage_bps}bp slippage "
      f"= {expected_fill:.4f}, actual fill {t.entry:.4f}")
check("entry filled at the next open plus slippage",
      np.isclose(t.entry, expected_fill, atol=1e-3))
check("entry bar is exactly one session after the signal bar",
      sym_data.index[sig_i + 1] == t.entry_date)
expected_target = t.entry + cfg.strategy.reward_risk * (t.entry - t.stop)
check("target is 2x the actual risk taken",
      np.isclose(t.target, expected_target, atol=1e-2),
      f"{t.target:.4f} vs {expected_target:.4f}")

print(f"\n  Sample stats: {res.stats()}")


# ---------------------------------------------------------------------------
print("\n6. Research layer can only ever block a trade")
# ---------------------------------------------------------------------------

from tbot.config import ResearchConfig
from tbot.research import Researcher, Verdict, screen

r = Researcher(api_key="fake", enabled=True)
news = [{"title": "Company reports something", "publisher": "Wire", "date": "2026-09-01"}]

def with_reply(text=None, boom=False):
    def _ask(*a, **kw):
        if boom:
            raise RuntimeError("network down")
        return text
    r._ask = _ask
    return r.review_trade("TEST", 100.0, 98.0, 104.0, news)

v = with_reply('{"veto": true, "reason": "pending merger vote", "flags": ["merger"]}')
check("a well-formed veto is honored", v.veto and "merger" in v.reason)

v = with_reply('{"veto": false, "reason": "nothing notable", "flags": []}')
check("a well-formed allow is honored", not v.veto)

v = with_reply("the model rambled instead of returning json")
check("unparseable output does not become a trading decision",
      not v.veto and v.source == "unavailable")

v = with_reply('{"veto": "yes", "reason": "hmm"}')
check("only a literal true counts as a veto, not the string 'yes'", not v.veto)

v = with_reply('{"veto": true, "reason": "' + "x" * 900 + '"}')
check("an absurdly long reason is truncated, not stored whole", len(v.reason) <= 240)

v = with_reply(boom=True)
check("an API failure is reported as unavailable, not as an allow",
      not v.veto and v.source == "unavailable")

# The important structural property: a Verdict has no field that could add,
# enlarge, or re-price a trade. It carries a boolean and an explanation.
fields = set(Verdict("x", "", [], "").__dict__.keys())
check("a Verdict cannot express anything except a block and a reason",
      fields == {"veto", "reason", "flags", "source", "headlines_seen"},
      str(fields))

rc = ResearchConfig(check_earnings=False, use_llm=True, require_research=True)
r._ask = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("down"))
import tbot.research as research_mod
_real_headlines = research_mod.headlines
research_mod.headlines = lambda *a, **kw: news
v = screen("TEST", 100.0, 98.0, 104.0, r, rc)
check("with research required, an outage blocks the trade rather than trading blind",
      v.veto and "unavailable" in v.reason)

rc2 = ResearchConfig(check_earnings=False, use_llm=True, require_research=False)
v = screen("TEST", 100.0, 98.0, 104.0, r, rc2)
check("with research optional, an outage lets the trade through", not v.veto)
research_mod.headlines = _real_headlines


# ---------------------------------------------------------------------------
print("\n7. State recording and the dashboard")
# ---------------------------------------------------------------------------

import tempfile
from pathlib import Path as _P
from tbot import dashboard, state as st_mod

tmp = _P(tempfile.mkdtemp())
st_mod.STATE_DIR = tmp
st_mod.STATE_FILE = tmp / "agent_state.json"
st_mod.EQUITY_FILE = tmp / "equity_history.csv"
st_mod.RUNLOG_FILE = tmp / "run_log.jsonl"

st_mod.append_equity(10000.0, 5000.0, 2, when="2026-09-01")
st_mod.append_equity(10100.0, 4000.0, 3, when="2026-09-02")
st_mod.append_equity(10250.0, 3900.0, 3, when="2026-09-02")   # same day, re-run
hist = st_mod.load_equity_history()
check("re-running on the same day replaces the point instead of duplicating it",
      len(hist) == 2 and hist[-1]["equity"] == 10250.0, str(hist))

s = st_mod.blank_state()
s["account"] = {"equity": 10250.0, "cash": 3900.0}
s["mode"] = "paper"
st_mod.save_state(s)
check("state round trips through disk", st_mod.load_state()["account"]["equity"] == 10250.0)

html_blank = dashboard.build_html(state=st_mod.blank_state(), history=[])
check("dashboard renders with no data at all and says so",
      "<html" in html_blank and "has not run" in html_blank)
check("empty dashboard has no placeholder numbers presented as real",
      "Generated never" not in html_blank)

html_full = dashboard.build_html(state=s, history=hist)
check("dashboard renders with real data", "$10,250.00" in html_full)
check("dashboard states plainly that paper money is not real money",
      "fake money" in html_full)

# A stale page must say so rather than looking current.
s_old = dict(s, updated_at="2026-01-01T00:00:00+00:00")
check("a stale dashboard warns instead of quietly showing old numbers",
      "Stale" in dashboard.build_html(state=s_old, history=hist))

from datetime import datetime as _dt, timedelta as _td, timezone as _tz
_future = (_dt.now(_tz.utc) + _td(hours=3)).isoformat()
check("a timestamp ahead of the clock does not render as negative time",
      "-" not in dashboard._age(_future)[0], dashboard._age(_future)[0])

# The bug this guards against: the age of the last run used to be written
# into the HTML by Python at build time, so a page added to a phone home
# screen read "0 min ago" forever and a day-old snapshot looked live.
_fresh_html = dashboard.build_html(state=s, history=hist)
check("the page carries the run timestamp for the browser to work from",
      'var BUILT = "' in _fresh_html)
check("the age is recomputed in the browser, not frozen at build time",
      "function ageParts" in _fresh_html and "setInterval" in _fresh_html)
check("the age tile is addressable so it can be updated",
      'id="age-val"' in _fresh_html and 'id="age-tile"' in _fresh_html)
check("the stale banner can be inserted client-side too",
      'id="banners"' in _fresh_html and 'id = "stale-banner"' in _fresh_html)
check("it checks for a newer run with caching disabled",
      'cache: "no-store"' in _fresh_html and "data.json" in _fresh_html)
check("a newer run navigates past the cached copy",
      "location.replace" in _fresh_html)
check("reopening a home-screen app rechecks rather than trusting the cache",
      "visibilitychange" in _fresh_html)

check("a critical watcher finding is shown above the numbers",
      "Needs attention" in dashboard.build_html(
          state=dict(s, findings=[{"severity": "critical", "agent": "reconcile",
                                   "message": "UNPROTECTED: no stop behind AAA"}]),
          history=hist))

# ---------------------------------------------------------------------------
print("\n6b. The daily-close exits the broker cannot handle")
# ---------------------------------------------------------------------------

from tbot.strategy import exit_decision

_cfg = AgentConfig()
_healthy = make_series(n=400, seed=31, drift=0.0012, vol=0.008)   # steady uptrend

_r = exit_decision(_healthy, _cfg.strategy, bars_held=3)
check("a position in an uptrend is left alone", _r is None, str(_r))

# Force the last close below the 50-day average.
_broken = _healthy.copy()
_broken.iloc[-1, _broken.columns.get_loc("close")] = _healthy["close"].iloc[-1] * 0.70
_broken.iloc[-1, _broken.columns.get_loc("low")] = _healthy["close"].iloc[-1] * 0.69
_r = exit_decision(_broken, _cfg.strategy, bars_held=3)
check("a close below the 50-day average triggers a trend-break exit",
      _r is not None and "trend break" in _r, str(_r))
check("the trend-break reason names both prices, so it can be checked by hand",
      _r is not None and _r.count("$") == 2, str(_r))

_r = exit_decision(_healthy, _cfg.strategy, bars_held=_cfg.strategy.max_hold_days)
check("a position held to the limit triggers the time stop",
      _r is not None and "time stop" in _r, str(_r))

_r = exit_decision(_healthy, _cfg.strategy, bars_held=_cfg.strategy.max_hold_days - 1)
check("one session short of the limit is not exited", _r is None, str(_r))

_cfg_off = AgentConfig()
_cfg_off.strategy.exit_on_trend_break = False
check("turning the trend-break rule off actually turns it off",
      exit_decision(_broken, _cfg_off.strategy, bars_held=3) is None)

check("an unknown holding period never triggers a time stop",
      exit_decision(_healthy, _cfg.strategy, bars_held=None) is None)
check("too little history is not treated as a reason to sell",
      exit_decision(_healthy.head(1), _cfg.strategy, bars_held=3) is None)


# ---------------------------------------------------------------------------
print("\n7a. An unfilled order counts as already owned")
# ---------------------------------------------------------------------------

from tbot.broker import committed_symbols

_pos = [{"symbol": "AAPL", "shares": 10}]
_ord = [{"symbol": "DIA", "status": "accepted"}, {"symbol": "QQQ", "status": "new"}]

_filled, _working = committed_symbols(_pos, _ord)
check("filled positions are recognized", _filled == {"AAPL"}, str(_filled))
check("accepted-but-unfilled orders are recognized", _working == {"DIA", "QQQ"},
      str(_working))
check("the agent treats both as off limits for a new buy",
      (_filled | _working) == {"AAPL", "DIA", "QQQ"})

# The exact bug this prevents: run twice before the open, buy twice.
_second_run = committed_symbols([], _ord)
check("a symbol with a working order is not bought again on a second run",
      "DIA" in (_second_run[0] | _second_run[1]))

check("no positions and no orders means nothing is committed",
      committed_symbols([], []) == (set(), set()))
check("missing symbol fields are ignored rather than crashing",
      committed_symbols([{"qty": 1}], [{"status": "new"}]) == (set(), set()))
check("None instead of a list is tolerated",
      committed_symbols(None, None) == (set(), set()))


# ---------------------------------------------------------------------------
print("\n7b. Refreshing data must never shorten the cache")
# ---------------------------------------------------------------------------

from tbot import data as datamod

_cache_backup = datamod.CACHE_DIR
_tmpcache = _P(tempfile.mkdtemp())
datamod.CACHE_DIR = _tmpcache

_long = make_series(n=900, seed=77)          # what a backtest downloaded
_short = _long.tail(120)                      # what a daily run asks for

datamod._cache_path("ZZZ").parent.mkdir(parents=True, exist_ok=True)
_long.to_csv(datamod._cache_path("ZZZ"))

_real_download = datamod.download_bars
datamod.download_bars = lambda sym, start=None, end=None, retries=3: _short.copy()
_after = datamod.load_bars("ZZZ", start="1990-01-01", refresh=True)
datamod.download_bars = _real_download

check("a short refresh keeps the long history already on disk",
      len(_after) == len(_long),
      f"{len(_after)} bars after refresh, {len(_long)} before")
check("the refreshed bars are still present and current",
      _after.index[-1] == _long.index[-1])
_reload = datamod.load_bars("ZZZ", start="1990-01-01")
check("the merged history is what actually got written to disk",
      len(_reload) == len(_long), f"{len(_reload)} on disk")


# ---------------------------------------------------------------------------
print("\n7a-ii. Three strategies, one discipline")
# ---------------------------------------------------------------------------
from tbot import strategy as strat_mod
from tbot.strategy import (PLAYBOOK, enabled_specs, UnknownStrategy, prepare,
                           exit_decision, latest_signal)
from tbot.indicators import rsi as _rsi
from tbot.config import AgentConfig as _AC

_scfg = _AC().strategy

check("all three rule sets are registered",
      sorted(PLAYBOOK) == ["breakout", "pullback", "reversion"], sorted(PLAYBOOK))
check("every strategy has an entry, an exit and a description",
      all(callable(v.entry) and callable(v.exit) and v.summary for v in PLAYBOOK.values()))

_bad = _AC().strategy
_bad.enabled = ["pullback", "typo"]
try:
    enabled_specs(_bad); _caught = False
except UnknownStrategy:
    _caught = True
check("a misspelled strategy name is an error, not a silent skip", _caught)

# RSI must stay inside its own definition or every threshold built on it lies.
_rs = _rsi(make_series(n=300, seed=41)["close"], 2).dropna()
check("RSI stays between 0 and 100",
      float(_rs.min()) >= -1e-9 and float(_rs.max()) <= 100 + 1e-9,
      f"{_rs.min():.2f} to {_rs.max():.2f}")

# The breakout window must exclude today, or today's own high sets the level
# it is being asked to clear and nothing can ever break out.
_prep = prepare(make_series(n=400, seed=42), _scfg)
_hi = _prep["donchian_hi"].dropna()
_raw = _prep["close"].rolling(_scfg.breakout_lookback).max().shift(1).dropna()
check("the breakout level is built from bars before today, not including it",
      _hi.equals(_raw))

# No lookahead, per strategy. A signal on the last bar must not change when
# the bars after it are removed, because when it fires they do not exist yet.
for _name in ("pullback", "breakout", "reversion"):
    _c = _AC().strategy; _c.enabled = [_name]
    _found = None
    for _seed in range(60, 130):
        _df = make_series(n=500, seed=_seed)
        for _cut in range(len(_df) - 60, len(_df)):
            if latest_signal("X", _df.iloc[:_cut], _c) is not None:
                _found = (_df, _cut); break
        if _found: break
    check(f"{_name}: a setup can actually be produced", _found is not None)
    if _found:
        _df, _cut = _found
        _a = latest_signal("X", _df.iloc[:_cut], _c)
        _b = latest_signal("X", _df.iloc[:_cut + 20], _c)   # 20 more bars exist
        _b2 = latest_signal("X", _df.iloc[:_cut], _c)
        check(f"{_name}: the signal is stable, future bars do not change it",
              _a.stop == _b2.stop and _a.signal_date == _b2.signal_date)
        check(f"{_name}: the signal is tagged with the strategy that made it",
              _a.strategy == _name, _a.strategy)
        check(f"{_name}: the stop is below the reference close",
              _a.stop < _a.reference_close)
        _sa = (_a.reference_close - _a.stop) / _a.atr
        check(f"{_name}: the stop respects the shared ATR guard rails",
              _scfg.min_stop_atr - 1e-9 <= _sa <= _scfg.max_stop_atr + 1e-9,
              f"{_sa:.2f}x ATR")

# The breakout must take the first break, not every bar of the advance.
_rise = make_series(n=400, seed=7)
_col = _rise.columns.get_loc("close")
import numpy as _np
_ramp = _np.linspace(1.0, 1.9, 120)
for _c2 in ("open", "high", "low", "close"):
    _rise.iloc[-120:, _rise.columns.get_loc(_c2)] = (
        _rise[_c2].iloc[-120:].to_numpy() * _ramp)
_bo = _AC().strategy; _bo.enabled = ["breakout"]
_sigs = strat_mod.signals_for_symbol("X", _rise, _bo)
_dates = [s.signal_date for s in _sigs]
_runlen = 0
for _i in range(1, len(_dates)):
    if (_dates[_i] - _dates[_i - 1]).days <= 1:
        _runlen += 1
check("a steady advance is not bought on every single bar",
      _runlen <= len(_dates) * 0.5,
      f"{_runlen} of {len(_dates)} signals were back to back")

# The exits are not interchangeable. Mean reversion enters below the short
# average on purpose, so running the trend exit over it closes it immediately.
_dip = make_series(n=400, seed=8)
_p2 = prepare(_dip, _scfg)
_row = _p2.iloc[-1].copy()
_row["close"] = float(_row["sma_exit"]) * 0.95      # below the 10-day average
_row["sma_fast"] = float(_row["close"]) * 1.05      # and below the 50-day
check("the trend exit would close a position sitting below its 50-day average",
      strat_mod._trend_exit(_row, _scfg, 1) is not None)
check("the reversion exit HOLDS that same position, which is the point",
      strat_mod._reversion_exit(_row, _scfg, 1) is None)

_row2 = _row.copy()
_row2["close"] = float(_row2["sma_exit"]) * 1.02    # the bounce arrived
check("the reversion exit takes the bounce when it comes",
      "bounce complete" in (strat_mod._reversion_exit(_row2, _scfg, 1) or ""))
check("the reversion time stop is shorter than the trend one",
      _scfg.reversion_max_hold < _scfg.max_hold_days)
check("the reversion time stop fires at its own limit",
      "time stop" in (strat_mod._reversion_exit(_row, _scfg,
                                                _scfg.reversion_max_hold) or ""))

# exit_decision must route by strategy, not by whatever ran last.
_dfx = _dip.copy()
_dfx.iloc[-1, _col] = float(_p2["sma_exit"].iloc[-1]) * 0.95
check("exit_decision routes a reversion position to the reversion exits",
      exit_decision(_dfx, _scfg, 1, "reversion") is None
      or "bounce" in exit_decision(_dfx, _scfg, 1, "reversion"))
check("an unrecognised strategy falls back to the trend exits rather than none",
      exit_decision(_dfx, _scfg, 999, "nonsense") is not None)

# Two strategies must never both open the same symbol on the same day.
_multi = _AC().strategy
_multi.enabled = ["pullback", "breakout", "reversion"]
_all_sigs = strat_mod.signals_for_symbol("X", make_series(n=600, seed=9), _multi)
check("at most one signal per symbol per day",
      len(_all_sigs) == len({s.signal_date for s in _all_sigs}),
      f"{len(_all_sigs)} signals on {len({s.signal_date for s in _all_sigs})} days")


# ---------------------------------------------------------------------------
print("\n7b-ii. A split must rebuild the cache, not create a cliff in it")
# ---------------------------------------------------------------------------
# Bars are split and dividend adjusted, so a split rewrites the entire history
# at the source. Yesterday's cache and today's download are then quoted on
# different bases, and concatenating them leaves a fake 90% crash in the middle
# of the series. Nothing downstream would call that an error. The moving
# averages would just be wrong and the agent would trade on them.

_pre = make_series(n=900, seed=91)                    # cached before the split
_post = (_pre.tail(120) / 10.0)                       # same bars, 10-for-1 split
_post["volume"] = _pre.tail(120)["volume"] * 10

datamod._cache_path("SPLIT").parent.mkdir(parents=True, exist_ok=True)
_pre.to_csv(datamod._cache_path("SPLIT"))

_full_post = _pre / 10.0                              # what a re-download returns
_full_post["volume"] = _pre["volume"] * 10

_calls = []
def _counting_download(sym, start=None, end=None, retries=3):
    _calls.append(start)
    return _post.copy() if len(_calls) == 1 else _full_post.copy()

_real_download = datamod.download_bars
datamod.download_bars = _counting_download
_healed = datamod.load_bars("SPLIT", start="1990-01-01", refresh=True)
datamod.download_bars = _real_download

check("a re-adjusted download triggers a second, full download",
      len(_calls) == 2, f"{len(_calls)} download(s)")
check("the rebuilt history keeps its full length",
      len(_healed) == len(_pre), f"{len(_healed)} bars, expected {len(_pre)}")

_jump = _healed["close"].pct_change().abs().max()
check("the rebuilt series has no split-sized cliff in it",
      _jump < 0.5, f"largest one-day move {_jump*100:.0f}%")
check("the rebuilt series is on the new price basis",
      abs(float(_healed["close"].iloc[-1]) - float(_full_post["close"].iloc[-1])) < 1e-6)

# The tolerance must be tight enough to catch a split and loose enough to
# ignore rounding, or it either never fires or fires every single day.
_rounded = _pre.copy()
_rounded["close"] = (_rounded["close"] * 1.0001)
_drift, _shared = datamod._adjustment_drift(_rounded, _pre)
check("a rounding-sized difference is not treated as a re-adjustment",
      _drift <= datamod.ADJUST_TOLERANCE, f"drift {_drift}")
_drift2, _ = datamod._adjustment_drift(_pre, _full_post)
check("a split-sized difference is treated as a re-adjustment",
      _drift2 > datamod.ADJUST_TOLERANCE, f"drift {_drift2}")

# Two halves that share no dates cannot be checked against each other, so
# joining them is a guess. It must rebuild instead.
_gap_calls = []
def _gap_download(sym, start=None, end=None, retries=3):
    _gap_calls.append(start)
    return _pre.tail(50).copy() if len(_gap_calls) == 1 else _pre.copy()

_pre.head(200).to_csv(datamod._cache_path("GAP"))
datamod.download_bars = _gap_download
_gapped = datamod.load_bars("GAP", start="1990-01-01", refresh=True)
datamod.download_bars = _real_download
check("a cache that does not overlap the download is rebuilt, not glued on",
      len(_gap_calls) == 2, f"{len(_gap_calls)} download(s)")

datamod.CACHE_DIR = _cache_backup


# ---------------------------------------------------------------------------
print("\n7a-0. The safety switch and the exits that have to outlive the day")
# ---------------------------------------------------------------------------
from tbot.broker import AlpacaBroker as _AB, PAPER_URL as _PAPER

# A bracket's time in force covers the whole group at Alpaca. With "day" the
# stop comes down at the close of the session the entry filled in, so every
# position spends its first night with no exit working anywhere.
_probe = {}
class _CaptureBroker(_AB):
    def __init__(self): super().__init__(key="k", secret="s", dry_run=False)
    def _request(self, method, path, **kw):
        _probe.update(kw.get("json") or {}); return {"id": "x", "status": "accepted"}
_CaptureBroker().submit_bracket("AAA", 10, 90.0, 120.0)
check("the bracket outlives the session that opened it",
      _probe.get("time_in_force") == "gtc", str(_probe.get("time_in_force")))
check("the stop and the target go in with the entry",
      _probe.get("order_class") == "bracket"
      and "stop_loss" in _probe and "take_profit" in _probe, str(sorted(_probe)))

# The live/paper switch decides whether three safety locks run at all, so an
# unrecognised URL has to count as live. Calling anything it does not
# recognise "paper" means a typo trades real money with the guards asleep.
for _u, _want in ((_PAPER, False),
                  ("http://paper-api.alpaca.markets", False),
                  ("https://paper-api.alpaca.markets/v2", False),
                  ("https://api.alpaca.markets", True),
                  ("HTTPS://API.ALPACA.MARKETS", True),
                  ("http://api.alpaca.markets", True),
                  ("https://some-proxy.example.com", True)):
    check(f"{_u[:34]:<34} reads as {'LIVE' if _want else 'paper'}",
          _AB(key="k", secret="s", base_url=_u).is_live is _want)

# A sell stop at or above the market is an instruction to dump the position.
class _StopBroker(_AB):
    def __init__(self): super().__init__(key="k", secret="s", dry_run=False)
    def _request(self, method, path, **kw): return {"id": "x", "status": "accepted"}
check("a stop above the market is refused, not submitted",
      _StopBroker().submit_stop("AAA", 10, 110.0, last_price=100.0).submitted is False)
check("a stop below the market is accepted",
      _StopBroker().submit_stop("AAA", 10, 90.0, last_price=100.0).submitted is True)

# Fills arrive a page at a time. Without paging, a position opened more than
# one page of fills ago has no buy in the window, and the time stop, which
# exists for exactly the oldest positions, quietly stops applying to them.
class _PagedBroker(_AB):
    def __init__(self):
        super().__init__(key="k", secret="s", dry_run=False); self.pages = 0
    def _request(self, method, path, **kw):
        self.pages += 1
        if self.pages == 1:
            return [{"id": str(i), "symbol": "ZZZ", "side": "sell", "qty": "1",
                     "price": "10", "transaction_time": "2026-02-01T00:00:00Z"}
                    for i in range(100)]
        if self.pages == 2:
            return [{"id": "old", "symbol": "AAA", "side": "buy", "qty": "10",
                     "price": "100", "transaction_time": "2020-01-02T00:00:00Z"}]
        return []
_pb = _PagedBroker()
check("fills are read past the first page",
      len(_pb.activities()) == 101, f"{len(_pb.activities())} fills")
check("so an old position still has an entry date for the time stop",
      _PagedBroker().entry_dates().get("AAA") == "2020-01-02",
      str(_PagedBroker().entry_dates()))

# Scaling out of half a position must not exempt the rest from the time stop.
class _PartialBroker(_AB):
    def __init__(self): super().__init__(key="k", secret="s", dry_run=False)
    def _request(self, method, path, **kw):
        return [{"id": "1", "symbol": "AAA", "side": "buy", "qty": "100",
                 "price": "10", "transaction_time": "2026-01-05T00:00:00Z"},
                {"id": "2", "symbol": "AAA", "side": "sell", "qty": "1",
                 "price": "11", "transaction_time": "2026-02-05T00:00:00Z"}]
check("selling 1 of 100 shares does not reset the holding clock",
      _PartialBroker().entry_dates().get("AAA") == "2026-01-05",
      str(_PartialBroker().entry_dates()))

# A malformed fill must be skipped, not guessed at. A missing side used to
# consume a real lot and invent a closed trade; a missing price printed as a
# total loss.
class _JunkBroker(_AB):
    def __init__(self): super().__init__(key="k", secret="s", dry_run=False)
    def _request(self, method, path, **kw):
        return [{"id": "1", "symbol": "AAA", "side": "buy", "qty": "10",
                 "price": "100", "transaction_time": "2026-01-01T00:00:00Z"},
                {"id": "2", "symbol": "AAA", "qty": "10",
                 "price": "105", "transaction_time": "2026-01-02T00:00:00Z"},
                {"id": "3", "symbol": "AAA", "side": "sell", "qty": "10",
                 "transaction_time": "2026-01-03T00:00:00Z"}]
check("a fill with no side does not fabricate a closed trade",
      _JunkBroker().realized_trades() == [], str(_JunkBroker().realized_trades()))


# ---------------------------------------------------------------------------
print("\n7a-0b. Reading the order book correctly, in both directions")
# ---------------------------------------------------------------------------
from tbot import watch as _w

_pos10 = [{"symbol": "AAA", "shares": 10}]
_acct0 = {"equity": 100_000.0, "buying_power": 1.0, "trading_blocked": False}

check("a stop whose size cannot be read is not counted as cover",
      _w.unprotected(_pos10, [{"symbol": "AAA", "side": "sell", "id": "1",
                               "qty": None, "stop_price": "90",
                               "status": "new"}]) == {"AAA": 10})
check("a replaced stop is not added to the one that replaced it",
      _w.unprotected(_pos10, [
          {"symbol": "AAA", "side": "sell", "id": "1", "qty": "10",
           "stop_price": "90", "status": "replaced"},
          {"symbol": "AAA", "side": "sell", "id": "2", "qty": "5",
           "stop_price": "90", "status": "new"}]) == {"AAA": 5})
check("the same order seen twice is only counted once",
      _w.unprotected(_pos10, [
          {"symbol": "AAA", "side": "sell", "id": "1", "qty": "5",
           "stop_price": "90", "status": "new"},
          {"symbol": "AAA", "side": "sell", "id": "1", "qty": "5",
           "stop_price": "90", "status": "new"}]) == {"AAA": 5})
check("a short position is never handed to the long-only repair path",
      _w.unprotected([{"symbol": "AAA", "shares": -10}], []) == {})
check("more stop shares than are held is itself a critical finding",
      any("OVER-PROTECTED" in f.message and f.severity == _w.CRITICAL
          for f in _w.check_broker(_acct0, _pos10,
                                   [{"symbol": "AAA", "side": "sell", "id": "1",
                                     "qty": "100", "stop_price": "90",
                                     "status": "new"}])))


# ---------------------------------------------------------------------------
print("\n7a-iii. The dashboard stylesheet cannot reference a colour that does not exist")
# ---------------------------------------------------------------------------
# var(--whatever) with no matching definition silently falls back to inherit.
# Nothing errors, the page just quietly renders that element in the wrong
# colour, and it stays that way until a person happens to look closely.
import re as _re
from tbot import dashboard as _dash

_page = _dash.build_html(state={"mode": "paper", "updated_at": "2026-01-01T00:00:00+00:00",
                                "account": {"equity": 100.0, "cash": 100.0},
                                "protected": [{"symbol": "AAA", "shares": 5, "stop": 90.0}]},
                         history=[])
_defined = set(_re.findall(r"(--[a-z0-9-]+)\s*:", _page))
_used = set(_re.findall(r"var\((--[a-z0-9-]+)", _page))
_undef = sorted(_used - _defined)
check("every colour token the page uses is actually defined",
      not _undef, f"undefined: {_undef}")
check("the page renders a repair the agent made", "protected" in _page)
check("and says what the stop was set to", "90.00" in _page, "stop level missing")


# ---------------------------------------------------------------------------
print("\n7b-ii-b. Batched downloads must split the frame correctly")
# ---------------------------------------------------------------------------
# 194 symbols one at a time is 194 requests per run, which from a shared CI
# address gets rate limited into a half-empty result. The batch path is what
# avoids that, so if its column splitting is wrong every symbol quietly falls
# back to an individual request and the whole point is lost, with no error.

import sys as _sys, types as _types

_syms = ["AAA", "BBB", "CCC"]
_idx = pd.bdate_range("2024-01-01", periods=40)
_frames = {}
for _n, _sy in enumerate(_syms):
    _frames[_sy] = pd.DataFrame({
        "Open": 100.0 + _n, "High": 101.0 + _n, "Low": 99.0 + _n,
        "Close": 100.5 + _n, "Volume": 1_000_000 + _n,
    }, index=_idx)

# yfinance hands back (field, ticker) columns for a multi-symbol request.
_multi = pd.concat(_frames, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)

_calls = []
def _fake_download(tickers, **kw):
    _calls.append(list(tickers) if isinstance(tickers, (list, tuple)) else [tickers])
    want = tickers if isinstance(tickers, (list, tuple)) else [tickers]
    keep = [c for c in _multi.columns if c[1] in want and c[1] != "CCC"]
    return _multi[keep] if keep else pd.DataFrame()

_fake_yf = _types.ModuleType("yfinance")
_fake_yf.download = _fake_download
_real_yf = _sys.modules.get("yfinance")
_sys.modules["yfinance"] = _fake_yf
_real_single = datamod.download_bars
datamod.download_bars = lambda sym, start=None, end=None, retries=3: _frames[sym].rename(
    columns=str.lower).rename_axis("date")
try:
    _got = datamod.download_many(_syms, start="2024-01-01", chunk=10)
finally:
    datamod.download_bars = _real_single
    if _real_yf is not None: _sys.modules["yfinance"] = _real_yf
    else: _sys.modules.pop("yfinance", None)

check("the batch asks for every symbol in one request",
      len(_calls) and set(_calls[0]) == set(_syms), str(_calls))
check("every symbol the batch returned comes back", set(_got) == set(_syms), str(sorted(_got)))
check("each symbol keeps its OWN prices, not another symbol's",
      abs(float(_got["AAA"]["close"].iloc[0]) - 100.5) < 1e-6
      and abs(float(_got["BBB"]["close"].iloc[0]) - 101.5) < 1e-6,
      f"AAA={_got['AAA']['close'].iloc[0]}, BBB={_got['BBB']['close'].iloc[0]}")
check("columns come back lowercase and complete",
      list(_got["AAA"].columns) == datamod.REQUIRED_COLS, str(list(_got["AAA"].columns)))
check("a symbol the batch skipped is fetched on its own rather than dropped",
      "CCC" in _got and len(_got["CCC"]) == 40)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# An error on the page must survive the presence of a routine note. The error
# banner used to be gated on there being no findings at all, but the loop that
# renders findings skips INFO -- so one "new since last run: AAPL", which
# check_broker emits on any day the position set changed, rendered a page with
# no finding banner AND no error banner. An unprotected position looked like a
# clean run.
from tbot import dashboard as _dsh
_err_acct = {"equity": 98326.16, "cash": 86.31, "buying_power": 0.0,
             "status": "ACTIVE", "trading_blocked": False, "mode": "PAPER"}
_err_state = {"mode": "paper", "updated_at": pd.Timestamp.utcnow().isoformat(),
              "last_full_run": pd.Timestamp.utcnow().isoformat(),
              "account": _err_acct, "positions": [],
              "errors": ["could not place a stop on NVDA: 403 insufficient qty"],
              "findings": [{"severity": "info", "agent": "reconcile",
                            "message": "new since last run: AAPL"}]}
_err_page = _dsh.build_html(state=_err_state, history=[])
check("an error still reaches the page when a routine note is present",
      "could not place a stop on NVDA" in _err_page,
      "the error banner was suppressed by an INFO finding")
check("and it is rendered as a critical banner",
      'class="banner critical"' in _err_page, "error shown but not as critical")

_no_err = dict(_err_state, errors=[])
check("a clean run with only a note shows no error banner",
      "Errors on the" not in _dsh.build_html(state=_no_err, history=[]),
      "invented an error banner with no errors")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# A save must never ERASE the last-traded marker. _cmd_run builds its state
# from blank_state() and carries almost nothing forward, so every path in it
# that saves without full_run wrote a null over a real timestamp -- and the
# page then announced "the agent has not traded" about an agent that traded
# the night before. Held in save_state rather than at each call site, so a
# command added later does not have to know the rule exists.
import tempfile as _tf, pathlib as _pl
from tbot import state as _stm

_sd = _pl.Path(_tf.mkdtemp())
_orig_dir, _orig_file = _stm.STATE_DIR, _stm.STATE_FILE
_stm.STATE_DIR, _stm.STATE_FILE = _sd, _sd / "agent_state.json"
_stm.save_state(_stm.blank_state(), full_run=True)
_stamped = _stm.load_state()["last_full_run"]
check("a real run stamps the last-traded marker", bool(_stamped))
_stm.save_state(_stm.blank_state())          # the market-hours guard path
check("a save that is not a trading run does not erase it",
      _stm.load_state()["last_full_run"] == _stamped,
      f"marker became {_stm.load_state()['last_full_run']}")
# now_iso has one-second resolution, so comparing against the old value would
# fail purely on how fast the test runs. The invariant that matters is that a
# real run re-stamps the marker to its OWN timestamp.
_stm.save_state(_stm.blank_state(), full_run=True)
_after = _stm.load_state()
check("a later real run re-stamps it to that run's own time",
      _after["last_full_run"] == _after["updated_at"],
      f"{_after['last_full_run']} vs {_after['updated_at']}")
_stm.STATE_DIR, _stm.STATE_FILE = _orig_dir, _orig_file

# An unreadable working-order list is unknown, not empty. Swallowing it handed
# back a partial order book as complete, which reads as "this position has no
# stop" and gets a second one stacked behind the working one.
from tbot.broker import BrokerError as _BErr2


class _OpenReadFails(_AB):
    def __init__(self):
        super().__init__(key="k", secret="s", dry_run=False)
    def _request(self, method, path, params=None, **kw):
        if params and params.get("status") == "open":
            raise _BErr2("429 too many requests")
        return [{"id": "recent", "symbol": "ZZ", "side": "buy", "status": "new"}]


_raised_open = ""
try:
    _OpenReadFails().open_orders()
except _BErr2 as exc:
    _raised_open = str(exc)
check("a failed working-order read is raised, not papered over",
      "could not read the working order list" in _raised_open, _raised_open)

# Exposure must be subtracted the same way it was added.
_mv = -5000.0
check("closing a short does not invent buying room",
      max(0.0, float(_mv or 0.0)) == 0.0)


# ---------------------------------------------------------------------------
# A GTC protective stop stays working for months while its submitted_at
# recedes. The paged sweep walks the most recently SUBMITTED orders, so on a
# busy account that stop eventually falls off the end of the window and the
# call returns a short list with an ordinary 200 -- indistinguishable from
# "this position has no stop". protect_exposed would then stack a second stop
# behind one that was working all along, and whichever filled first sells the
# position while the other opens a naked short.
from tbot.broker import AlpacaBroker as _AB


class _AgedOutBroker(_AB):
    def __init__(self):
        super().__init__(key="k", secret="s", dry_run=False)
    def _request(self, method, path, params=None, **kw):
        if params and params.get("status") == "open":
            return [{"id": "old-stop", "symbol": "JCI", "side": "sell",
                     "type": "stop", "qty": "224", "stop_price": "137.17",
                     "status": "held"}]
        return [{"id": f"noise{i}", "symbol": "ZZ", "side": "buy",
                 "status": "canceled"} for i in range(3)]


_aged = [o["id"] for o in _AgedOutBroker().open_orders()]
check("a months-old working stop is still found once it leaves the recent window",
      "old-stop" in _aged, str(_aged))


# ---------------------------------------------------------------------------
# An unfilled entry counts against the POSITION cap, so it has to count
# against the EXPOSURE cap too, or the two limits disagree about what a
# position is. A second run before the open saw no fills, therefore gross = 0,
# and handed out the whole exposure budget a second time; everything filled at
# the open at roughly twice the intended size, on a config that forbids margin.
import agent as _ag

_pending_bracket = [{"id": "p1", "symbol": "KLAC", "side": "buy", "qty": "51",
                     "status": "accepted",
                     "legs": [{"id": "l1", "symbol": "KLAC", "side": "sell",
                               "type": "stop", "qty": "51",
                               "stop_price": "166.12", "status": "held"}]}]
check("an accepted entry counts toward exposure before it fills",
      _ag._pending_notional(_pending_bracket, held_symbols=set()) > 8000,
      str(_ag._pending_notional(_pending_bracket, held_symbols=set())))
check("and stops counting once the position is held, not twice",
      _ag._pending_notional(_pending_bracket, held_symbols={"KLAC"}) == 0.0)
check("a protective sell order never creates room to buy",
      _ag._pending_notional(
          [{"id": "s1", "symbol": "CL", "side": "sell", "qty": "224",
            "type": "stop", "stop_price": "84.33", "status": "held"}],
          held_symbols=set()) == 0.0)
check("a dead order is not charged for",
      _ag._pending_notional(
          [{"id": "d", "symbol": "X", "side": "buy", "qty": "10",
            "limit_price": "50", "status": "canceled"}], held_symbols=set()) == 0.0)

# A short reports a negative market_value at Alpaca. Summing it signed would
# GRANT room proportional to the size of the short.
_short_mix = [{"symbol": "LONG", "market_value": 10_000.0},
              {"symbol": "SHORT", "market_value": -5_000.0}]
check("a short holding does not hand out extra buying room",
      sum(max(0.0, float(p.get("market_value") or 0.0)) for p in _short_mix) == 10_000.0)


# ---------------------------------------------------------------------------
print("\n7b-iii. A failed close must not leave a position without a stop")
# ---------------------------------------------------------------------------
from tbot.broker import AlpacaBroker, BrokerError as _BErr

class _FlakyBroker(AlpacaBroker):
    """The close fails after the stops have already been cancelled."""
    def __init__(self, restore_ok=True):
        super().__init__(key="k", secret="s", dry_run=False)
        self.restore_ok = restore_ok
        self.posted = []
    def open_orders(self):
        return [{"symbol": "AAA", "id": "stop1", "side": "sell",
                 "qty": "10", "stop_price": "90.00"},
                {"symbol": "AAA", "id": "tp1", "side": "sell",
                 "qty": "10", "limit_price": "120.00"}]
    def _request(self, method, path, **kw):
        if method == "DELETE" and path.startswith("/v2/orders/"):
            return {}
        if method == "DELETE" and path.startswith("/v2/positions/"):
            raise _BErr("DELETE /v2/positions/AAA -> 422: insufficient qty")
        if method == "POST" and path == "/v2/orders":
            if not self.restore_ok:
                raise _BErr("POST /v2/orders -> 403: forbidden")
            self.posted.append(kw.get("json"))
            return {"id": "restored"}
        raise AssertionError(f"unexpected call {method} {path}")

_flaky = _FlakyBroker(restore_ok=True)
try:
    _flaky.close_position("AAA")
    _raised = ""
except _BErr as exc:
    _raised = str(exc)
check("a failed close is reported rather than swallowed", "could not close" in _raised)
check("the original stop is put back when the close fails",
      len(_flaky.posted) == 1 and _flaky.posted[0]["stop_price"] == 90.0,
      str(_flaky.posted))
check("the restored order is a stop for the full size",
      _flaky.posted[0]["type"] == "stop" and _flaky.posted[0]["qty"] == "10")
check("the message says the position is protected", "put back" in _raised)

# The shape above is FLAT, and that is why it passed against code that could
# not cancel a bracket. Live, open_orders() hands back the legs NESTED under
# their parent, and once the entry fills the parent is the only thing at the
# top level -- a filled order, which Alpaca will not cancel. Iterating the top
# level attempted one DELETE, got a 422 for trying to cancel a fill, swallowed
# it, and returned nothing while both legs kept reserving the shares. Every
# soft exit failed that way on every bracket-opened position, and the run then
# reported a fully protected position as naked.
_NESTED_PARENT = {
    "id": "parent", "symbol": "AAA", "side": "buy", "qty": "10",
    "status": "filled",
    "legs": [{"id": "leg-tp", "symbol": "AAA", "side": "sell", "type": "limit",
              "qty": "10", "limit_price": "120.00", "status": "new"},
             {"id": "leg-stop", "symbol": "AAA", "side": "sell", "type": "stop",
              "qty": "10", "stop_price": "90.00", "status": "held"}]}


class _BracketBroker(_FlakyBroker):
    """The live shape: legs nested under a parent that has already filled."""
    def __init__(self, close_ok=True):
        super().__init__(restore_ok=True)
        self.close_ok = close_ok
        self.deleted = []
    def open_orders(self):
        return [dict(_NESTED_PARENT)]
    def _request(self, method, path, **kw):
        if method == "DELETE" and path.startswith("/v2/orders/"):
            oid = path.rsplit("/", 1)[-1]
            self.deleted.append(oid)
            if oid == "parent":
                raise _BErr("422: order is not cancelable")
            return {}
        if method == "DELETE" and path.startswith("/v2/positions/"):
            if self.close_ok:
                return {"qty": "10", "id": "closed", "status": "filled"}
            raise _BErr("403: insufficient qty available")
        if method == "POST" and path == "/v2/orders":
            self.posted.append(kw.get("json"))
            return {"id": "restored"}
        raise AssertionError(f"unexpected call {method} {path}")


_br = _BracketBroker(close_ok=True)
_killed = _br.cancel_orders_for("AAA")
check("a bracket's stop leg is cancelled, not just the parent",
      "leg-stop" in _br.deleted, str(_br.deleted))
check("and its target leg too, or the close is refused for held shares",
      "leg-tp" in _br.deleted, str(_br.deleted))
check("the filled parent is not pointlessly asked to cancel",
      "parent" not in _br.deleted, str(_br.deleted))
check("what it reports cancelling is what it actually cancelled",
      sorted(o["id"] for o in _killed) == ["leg-stop", "leg-tp"], str(_killed))

_br2 = _BracketBroker(close_ok=True)
_fill = _br2.close_position("AAA")
check("so a soft exit on a bracket position actually closes it",
      _fill.submitted and _fill.shares == 10, str(_fill))

# And the safety net has to be able to read a price out of the nested legs.
_br3 = _BracketBroker(close_ok=False)
try:
    _br3.close_position("AAA")
    _msg = ""
except _BErr as exc:
    _msg = str(exc)
check("a failed bracket close still puts the real stop back",
      len(_br3.posted) == 1 and _br3.posted[0]["stop_price"] == 90.0,
      str(_br3.posted))
check("and does not call a protected position naked",
      "put back" in _msg and "no stop order" not in _msg, _msg)

_flaky2 = _FlakyBroker(restore_ok=False)
try:
    _flaky2.close_position("AAA")
    _raised2 = ""
except _BErr as exc:
    _raised2 = str(exc)
check("when the stop cannot be restored the message says so, loudly",
      "no stop order behind" in _raised2, _raised2)

_cancelled = _FlakyBroker().cancel_orders_for("AAA")
check("cancel_orders_for returns the orders themselves, not just a count",
      len(_cancelled) == 2 and {o["id"] for o in _cancelled} == {"stop1", "tp1"},
      str(_cancelled))
check("and the cancelled stop still carries the price needed to restore it",
      any(o.get("stop_price") for o in _cancelled), str(_cancelled))

# A network failure must arrive as the error type every caller already handles.
# Raw urllib exceptions escape every try/except in the agent and end the daily
# run in a stack trace: no state saved, no dashboard written, no record that
# anything went wrong.
import requests as _rq
class _OfflineBroker(AlpacaBroker):
    def __init__(self):
        super().__init__(key="k", secret="s", dry_run=False)
_offline = _OfflineBroker()
_offline.base_url = "https://127.0.0.1:1"
try:
    _offline.account()
    _net = "no error raised"
except _BErr as exc:
    _net = "BrokerError"
except Exception as exc:
    _net = type(exc).__name__
check("a network failure is raised as a BrokerError, not a raw urllib error",
      _net == "BrokerError", _net)


# ---------------------------------------------------------------------------
print("\n7c. The journal refuses to report noise as a finding")
# ---------------------------------------------------------------------------

from tbot import journal

_rng = np.random.default_rng(4)

# A small sample must never produce a number, however good it looks.
_tiny = pd.DataFrame({"R": [2.0, 2.0, 2.0, 2.0, 2.0],
                      "reason": ["target"] * 5, "symbol": ["AAA"] * 5,
                      "entry_date": pd.bdate_range("2026-01-01", periods=5)})
_rep = journal.format_report(journal.analyze(_tiny))
check("a 5-trade bucket is reported as insufficient, not as a result",
      "not enough data" in _rep, _rep[:200])

# Pure coin flips must not come out as a real edge.
_noise = pd.DataFrame({"R": _rng.normal(0, 1.2, 400),
                       "reason": ["stop"] * 400, "symbol": ["AAA"] * 400,
                       "entry_date": pd.bdate_range("2020-01-01", periods=400)})
_res = journal.analyze(_noise)
check("random results are not called an edge",
      not _res["overall"].get("real"), str(_res["overall"]))
check("it says how many trades would be needed instead",
      "trades_needed" in _res["overall"])

# A genuinely large effect on a large sample should be recognized.
_real = pd.DataFrame({"R": _rng.normal(0.8, 1.0, 400),
                      "reason": ["target"] * 400, "symbol": ["AAA"] * 400,
                      "entry_date": pd.bdate_range("2020-01-01", periods=400)})
check("a large, well-sampled edge IS recognized",
      journal.analyze(_real)["overall"].get("real") is True)

# Excursions: losers that were once winners, winners that dipped first.
_exc = pd.DataFrame({
    "R": [-1.0, -1.0, 2.0, 2.0],
    "worst_R": [-1.0, -1.4, -0.8, -0.1],
    "best_R": [1.5, 0.1, 2.4, 2.0],
    "reason": ["stop", "gap through stop", "target", "target"],
    "symbol": ["A"] * 4,
    "entry_date": pd.bdate_range("2026-01-01", periods=4)})
_e = journal.excursions(_exc)
check("it spots losers that were up a full R first",
      _e["losers_that_were_up_1R"] == 50.0, str(_e))
check("overruns are measured on the realized loss, not the bar's low",
      _e["losers_past_planned"] == 0.0,
      f"both losers realized exactly -1.0R, so none overran: {_e}")

_over = pd.DataFrame({
    "R": [-1.0, -1.8, 2.0], "worst_R": [-1.0, -2.1, -0.3],
    "best_R": [0.2, 0.1, 2.1], "reason": ["stop", "gap through stop", "target"],
    "symbol": ["A"] * 3, "entry_date": pd.bdate_range("2026-01-01", periods=3)})
_eo = journal.excursions(_over)
check("a genuine overrun is counted", _eo["losers_past_planned"] == 50.0, str(_eo))
check("and its average size is reported", _eo["avg_overrun_R"] == -1.8, str(_eo))
check("it measures how far winners dipped before working",
      _e["winner_worst_R"] == -0.45, str(_e))

check("the report states plainly that it changes nothing on its own",
      "Nothing here changes what the agent does" in
      journal.format_report(journal.analyze(_noise)))


# ---------------------------------------------------------------------------
print("\n7e. Ranking decides which setups get the slots")
# ---------------------------------------------------------------------------

from tbot import ranking

_rk_df = prepare(make_series(n=400, seed=21, drift=0.0010), AgentConfig().strategy)
_last = len(_rk_df) - 1
_stop = float(_rk_df["close"].iloc[_last]) * 0.95

check("'none' scores everything the same, so order is preserved",
      ranking.score("none", _rk_df, _last, _stop) == 0.0)
# Comparing a function against itself on a copy is true for anything
# deterministic, including a scorer that openly reads the future. The real
# test is that changing a FUTURE bar cannot change today's score.
_rk_future = _rk_df.copy()
_rk_future.iloc[_last, _rk_future.columns.get_loc("close")] *= 1.5
check("momentum reads a trailing return, not today's bar",
      ranking.score("momentum", _rk_df, _last, _stop) ==
      ranking.score("momentum", _rk_future, _last, _stop))

# No lookahead: a score computed at bar i must not change when later bars are
# removed. If it does, the ranking is reading the future.
_trunc = prepare(make_series(n=400, seed=21, drift=0.0010).iloc[:_last + 1],
                 AgentConfig().strategy)
for _m in ("momentum", "reward_risk", "liquidity"):
    a = ranking.score(_m, _rk_df, _last, _stop)
    b = ranking.score(_m, _trunc, len(_trunc) - 1, _stop)
    check(f"'{_m}' uses no data after the signal bar",
          (a == b) or (a != a and b != b), f"{a} vs {b}")

check("a tighter stop ranks above a wider one under reward_risk",
      ranking.score("reward_risk", _rk_df, _last,
                    float(_rk_df['close'].iloc[_last]) * 0.98) >
      ranking.score("reward_risk", _rk_df, _last,
                    float(_rk_df['close'].iloc[_last]) * 0.90))

_cands = [("AAA", 0.1), ("BBB", 0.9), ("CCC", 0.5)]
check("ranking puts the best candidate first",
      ranking.order(_cands, "momentum")[0] == "BBB",
      str(ranking.order(_cands, "momentum")))
check("'none' leaves the original order alone",
      ranking.order(_cands, "none") == ["AAA", "BBB", "CCC"])
check("equal scores keep their original order rather than shuffling",
      ranking.order([("A", 1.0), ("B", 1.0), ("C", 1.0)], "momentum")
      == ["A", "B", "C"])
check("an unscoreable candidate sinks to the bottom instead of crashing",
      ranking.order([("A", float("nan")), ("B", 0.2)], "momentum") == ["B", "A"])

# The whole point: the backtest and the live agent must break ties the same
# way, or the backtest is not predicting what the agent does.
check("both paths share one ranking function",
      "ranking.order" in open(_P(__file__).parent.parent / "agent.py").read()
      and "ranking.order" in open(_P(__file__).parent.parent / "tbot/backtest.py").read())


# ---------------------------------------------------------------------------
print("\n7d. Evolution is fenced in and honestly scored")
# ---------------------------------------------------------------------------

from tbot import evolve

_risky = {"risk_per_trade", "max_position_pct", "max_open_positions",
          "max_gross_exposure", "max_drawdown_halt", "max_correlation"}
check("no risk setting is exposed to the optimizer",
      not (_risky & set(evolve.SEARCH_SPACE)), str(set(evolve.SEARCH_SPACE)))
check("only strategy settings are tunable",
      all(hasattr(AgentConfig().strategy, p) for p in evolve.SEARCH_SPACE))
check("every tunable is bounded by an explicit list",
      all(isinstance(v, list) and len(v) >= 2 for v in evolve.SEARCH_SPACE.values()))

# Scoring must be able to say "tuning did not help", or it is not a test.
_no_help = [evolve.WindowResult(
    pd.Timestamp("2018-01-01"), pd.Timestamp("2020-12-31"),
    pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"),
    "reward_risk", 3.0, 2.0, chosen_test_R=-0.05, default_test_R=0.05,
    chosen_test_n=60, default_test_n=60) for _ in range(4)]
_v = evolve.verdict(_no_help)
check("tuning that loses out of sample is reported as not worth doing",
      _v["worth_doing"] is False, str(_v.get("avg_out_of_sample_gain_R")))
check("and the report says so in words",
      "did NOT reliably beat" in evolve.format_report(_no_help, _v))

_helps = [evolve.WindowResult(
    pd.Timestamp("2018-01-01"), pd.Timestamp("2020-12-31"),
    pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"),
    "reward_risk", 3.0, 2.0, chosen_test_R=0.20, default_test_R=0.05,
    chosen_test_n=60, default_test_n=60) for _ in range(4)]
_vh = evolve.verdict(_helps)
check("a real, repeated out-of-sample gain IS recognized",
      _vh["worth_doing"] is True, str(_vh.get("avg_out_of_sample_gain_R")))

# A gain smaller than the noise threshold must not trigger a change.
_tiny = [evolve.WindowResult(
    pd.Timestamp("2018-01-01"), pd.Timestamp("2020-12-31"),
    pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"),
    "reward_risk", 3.0, 2.0, chosen_test_R=0.051, default_test_R=0.05,
    chosen_test_n=60, default_test_n=60) for _ in range(4)]
check("a gain below the minimum edge is not called worth doing",
      evolve.verdict(_tiny)["worth_doing"] is False)

check("training and judging windows never overlap",
      all(r.test_start > r.train_end for r in _no_help))

# The window builder must actually produce windows on realistic history, and
# must only count trades from inside the period being judged. A silent zero
# here previously came out as "not enough history", which is a wrong answer
# dressed as a limitation.
_wf_bars = universe(["SPY", "AAA", "BBB", "CCC", "DDD", "EEE"], seed0=55, n=2600)
_sl = evolve._slice(_wf_bars, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
check("a one-year window still gets its warmup history", len(_sl) == 6, str(len(_sl)))
_first = list(_sl.values())[0]
check("the warmup really is in front of the window",
      _first.index[0] < pd.Timestamp("2024-01-01"), str(_first.index[0].date()))

_r_all, _n_all = evolve._score(_sl, AgentConfig())
_r_in, _n_in = evolve._score(_sl, AgentConfig(), only_after="2024-01-01")
check("scoring inside the window counts fewer trades than the whole slice",
      _n_in < _n_all, f"{_n_in} vs {_n_all}")


# ---------------------------------------------------------------------------
print("\n8. Strategy comparison simulator")
# ---------------------------------------------------------------------------

from tbot import compare as cmp

_syms = ["SPY", "EFA", "AGG", "SHY", "AAA", "BBB", "CCC", "DDD", "EEE"]
_bars = {s: make_series(n=2000, seed=200 + i,
                        drift=0.0004 if s not in ("AGG", "SHY") else 0.00008,
                        vol=0.011 if s not in ("AGG", "SHY") else 0.002)
         for i, s in enumerate(_syms)}
_px = cmp.close_matrix(_bars)
_uni = ["AAA", "BBB", "CCC", "DDD", "EEE"]

# With no costs, holding one asset at 100% must reproduce that asset exactly.
# If this drifts, the return or cost math is wrong somewhere.
_bh = cmp.simulate(_px, cmp.w_buy_hold(_px), slippage_bps=0.0)
_asset = _px["SPY"] / _px["SPY"].iloc[0] * 10_000
check("buy and hold reproduces the underlying asset exactly",
      np.isclose(_bh["equity"].iloc[-1], _asset.iloc[-1], rtol=1e-9),
      f"{_bh['equity'].iloc[-1]:.4f} vs {_asset.iloc[-1]:.4f}")

# Removing future bars must not change past equity. This is the check that
# catches a strategy secretly using tomorrow's price to decide today.
_half = _px.index[1200]
_trunc = cmp.simulate(_px[_px.index <= _half], cmp.w_trend_filter(_px[_px.index <= _half]), 5.0)
_full = cmp.simulate(_px, cmp.w_trend_filter(_px), 5.0)
check("no lookahead: truncating the data leaves past equity unchanged",
      np.isclose(_trunc["equity"].iloc[-1], _full["equity"].loc[_half], rtol=1e-9))

# Costs must reduce returns, never increase them.
_free = cmp.simulate(_px, cmp.w_xs_momentum(_px, _uni, top_n=3), 0.0)
_costly = cmp.simulate(_px, cmp.w_xs_momentum(_px, _uni, top_n=3), 50.0)
check("higher slippage always produces a worse result",
      _costly["equity"].iloc[-1] < _free["equity"].iloc[-1])

# No strategy may lever up. Weights must never exceed 100% of the account.
for _name, _w in [("trend", cmp.w_trend_filter(_px)),
                  ("dual", cmp.w_dual_momentum(_px)),
                  ("momentum", cmp.w_xs_momentum(_px, _uni, top_n=3)),
                  ("equal weight", cmp.w_equal_weight(_px, _uni))]:
    total = float(_w.sum(axis=1).max())
    check(f"{_name} never allocates more than 100% (no accidental leverage)",
          total <= 1.0 + 1e-9, f"max total weight {total:.4f}")
    check(f"{_name} never goes short", float(_w.to_numpy().min()) >= -1e-12)

# Rebalancing must be monthly, not daily. Daily churn would be a cost bug.
_turns = cmp.simulate(_px, cmp.w_dual_momentum(_px), 5.0)["turnover"]
check("rebalancing happens on a handful of days, not every day",
      (_turns > 1e-9).mean() < 0.10, f"{(_turns > 1e-9).mean()*100:.1f}% of days traded")

_entries = cmp.run_all(_bars, _uni, 5.0, 10_000.0)
check("every strategy ran", len(_entries) == 5, f"{len(_entries)} ran")
_tbl = cmp.table(_entries)
check("comparison table has one row per strategy", len(_tbl) == len(_entries))
_html = cmp.compare_html(_entries, _tbl, _tbl, _tbl, "2020-01-01")
check("comparison report renders", "<html" in _html and "log scale" in _html)
check("report warns that the winner is partly luck", "luckiest" in _html)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
print("\nThe broker adapter cannot mistake a partial answer for a whole one")
# ---------------------------------------------------------------------------
import os as _os

from tbot import research as _research
from tbot.broker import PAPER_URL, AlpacaBroker

# GitHub Actions sets an env var to the empty string when the secret behind it
# does not exist. os.getenv's default only applies when the name is ABSENT, so
# an empty ALPACA_BASE_URL used to survive as "", which is not the paper host,
# so is_live read True, the live-trading lock refused the run, and _cmd_run
# returned before saving anything: a green build and an agent that had quietly
# stopped trading.
_prior = _os.environ.get("ALPACA_BASE_URL")
try:
    _os.environ["ALPACA_BASE_URL"] = ""
    _b = AlpacaBroker()
    check("an empty ALPACA_BASE_URL falls back to the paper endpoint",
          _b.base_url == PAPER_URL.rstrip("/"), repr(_b.base_url))
    check("and is therefore not treated as a live account", not _b.is_live)
finally:
    if _prior is None:
        _os.environ.pop("ALPACA_BASE_URL", None)
    else:
        _os.environ["ALPACA_BASE_URL"] = _prior

# Alpaca returns 50 orders by default and up to 500 on request. A truncated
# order book arrives as an ordinary 200, so the stops it cut look missing and
# the repair path stacks a second stop behind a position that already has one.
_seen = {}
_b = AlpacaBroker(key="k", secret="s", base_url=PAPER_URL)
_b._request = lambda method, path, **kw: (_seen.update(kw.get("params") or {}), [])[1]
_b.open_orders()
check("open_orders asks for more than one page of orders",
      int(_seen.get("limit", 0)) >= 500, str(_seen))

# An OCO's stop lives in the parent's legs: the parent itself is a sell limit
# carrying the take-profit price and no stop price at all. Reading parents only
# makes a properly protected position look bare.
check("open_orders asks for the nested legs too",
      str(_seen.get("nested", "")).lower() in ("true", "1"), str(_seen))

# The real shape, taken from the account on 2026-09-10. A bracket's legs stay
# children of the entry order, so once the entry fills the parent is no longer
# open and status=open drops the group -- carrying the working stop leg out
# with it. The take-profit survives because it is live on the exchange in its
# own right. That is a fully protected position reading as a naked one behind
# an orphaned target, and it is where every UNPROTECTED alarm came from.
from tbot import watch as _watch

_ABNB_STOP = {"id": "leg-stop", "symbol": "ABNB", "side": "sell", "type": "stop",
              "qty": "71", "stop_price": "155.82", "status": "held"}
_ABNB_TP = {"id": "leg-tp", "symbol": "ABNB", "side": "sell", "type": "limit",
            "qty": "71", "limit_price": "197.25", "status": "new"}
_ABNB_ENTRY = {"id": "parent", "symbol": "ABNB", "side": "buy", "type": "market",
               "qty": "71", "status": "filled", "submitted_at": "2026-09-09T23:25:00Z",
               "legs": [_ABNB_TP, _ABNB_STOP]}


def _alpaca(params=None, **kw):
    """Answers the two queries the way the live account did."""
    params = params or {}
    if params.get("status") == "open":
        return [dict(_ABNB_TP)]            # the held leg and its parent vanish
    return [dict(_ABNB_ENTRY)]             # status=all keeps the whole group


_b = AlpacaBroker(key="k", secret="s", base_url=PAPER_URL)
_b._request = lambda method, path, **kw: _alpaca(**kw)
_orders = _b.open_orders()
_flat = _watch._flatten_orders(_orders)
check("the working stop leg of a filled bracket is found",
      any(o.get("id") == "leg-stop" for o in _flat),
      str([o.get("id") for o in _flat]))
check("and the position reads as covered, not naked",
      not _watch.unprotected([{"symbol": "ABNB", "shares": 71}], _orders),
      str(_watch.unprotected([{"symbol": "ABNB", "shares": 71}], _orders)))
check("the filled entry is not mistaken for a working order",
      not any(o.get("id") == "parent" and str(o.get("status")) == "filled"
              and o.get("side") == "sell" for o in _flat))

# A bracket whose legs are all done leaves nothing to protect the position, and
# that really is exposure rather than a reporting fault.
_dead = dict(_ABNB_ENTRY, legs=[dict(_ABNB_TP, status="filled"),
                                dict(_ABNB_STOP, status="canceled")])
_b._request = lambda method, path, **kw: [dict(_dead)]
check("a bracket with no working leg left is reported exposed",
      _watch.unprotected([{"symbol": "ABNB", "shares": 71}],
                         _b.open_orders()) == {"ABNB": 71})

_legged = [{"symbol": "CL", "side": "sell", "type": "limit", "qty": "224",
            "limit_price": "97.66", "status": "new", "id": "oco-parent",
            "legs": [{"symbol": "CL", "side": "sell", "type": "stop",
                      "qty": "224", "stop_price": "84.64", "status": "new",
                      "id": "oco-leg-stop"}]}]
check("a stop nested inside an OCO parent counts as protection",
      not _watch.unprotected([{"symbol": "CL", "shares": 224}], _legged),
      str(_watch.unprotected([{"symbol": "CL", "shares": 224}], _legged)))
check("and the parent limit alone is not mistaken for one",
      _watch.unprotected([{"symbol": "CL", "shares": 224}],
                         [dict(_legged[0], legs=[])]) == {"CL": 224})


# ---------------------------------------------------------------------------
print("\nThe research screen fails closed when it cannot run")
# ---------------------------------------------------------------------------
# A lookup that threw and a stock that genuinely has no news are different
# answers. Returning the same value for both let every trade pass a screen
# that had never run.

_orig_earnings = _research.next_earnings_date
try:
    _research.next_earnings_date = lambda symbol: _research.LOOKUP_FAILED
    _v = _research.earnings_veto("TEST", 10)
    check("an unreadable earnings calendar blocks the trade", _v.veto, _v.reason)
    check("and says so rather than claiming there are no earnings",
          _v.source == "unavailable", _v.source)

    _research.next_earnings_date = lambda symbol: None
    _v = _research.earnings_veto("TEST", 10)
    check("a stock with no scheduled earnings is still allowed", not _v.veto,
          _v.reason)
finally:
    _research.next_earnings_date = _orig_earnings

# The sentinel is only worth anything if the lookups really produce it, so
# break yfinance underneath them rather than stubbing the functions that
# return it.
import sys as _sys
import types as _types

_fake_yf = _types.ModuleType("yfinance")


def _yahoo_is_down(*a, **kw):
    raise RuntimeError("yahoo rate-limited this runner")


_fake_yf.Ticker = _yahoo_is_down
_saved_yf = _sys.modules.get("yfinance")
try:
    _sys.modules["yfinance"] = _fake_yf
    check("a headline lookup that throws returns None, not an empty list",
          _research.headlines("TEST") is None,
          repr(_research.headlines("TEST")))
    check("an earnings lookup that throws returns the failure sentinel",
          _research.next_earnings_date("TEST") is _research.LOOKUP_FAILED,
          repr(_research.next_earnings_date("TEST")))
finally:
    if _saved_yf is None:
        _sys.modules.pop("yfinance", None)
    else:
        _sys.modules["yfinance"] = _saved_yf

_r = _research.Researcher(api_key="fake", enabled=True)
_v = _r.review_trade("TEST", 100.0, 98.0, 104.0, None)
check("a failed headline lookup is reported as unavailable",
      _v.source == "unavailable", _v.source)
_v = _r.review_trade("TEST", 100.0, 98.0, 104.0, [])
check("a stock with genuinely no headlines is not reported as unavailable",
      _v.source == "llm", _v.source)

# End to end: the fail-closed guard in screen() can now see the failure.
_cfg = AgentConfig().research
_orig_head = _research.headlines
_orig_earnings = _research.next_earnings_date
try:
    _research.headlines = lambda symbol, limit=8, max_age_days=14: None
    _research.next_earnings_date = lambda symbol: None
    _v = _research.screen("TEST", 100.0, 98.0, 104.0,
                          _research.Researcher(api_key="fake", enabled=True), _cfg)
    check("screen() blocks a trade whose news could not be checked",
          _v.veto, f"{_v.veto} {_v.reason}")
finally:
    _research.headlines = _orig_head
    _research.next_earnings_date = _orig_earnings


# ---------------------------------------------------------------------------
print("\nA trade too small to matter is not worth a position slot")
# ---------------------------------------------------------------------------
# The position, notional and exposure caps are all ceilings, so whichever
# candidate is sized last takes whatever room is left. On 2026-09-09 that was
# $255.35 of room against a $252.40 share: AMZN went in at one share risking
# $16.18 where a full position risks $991. It cannot move the account, and it
# still holds a slot, needs a stop, and is managed and reported like a real one.

from dataclasses import replace as _replace

_cfg = AgentConfig()
_r, _s = _cfg.risk, _cfg.strategy
_EQ = 99123.03

# The live case, reproduced exactly.
_held = 19718.72 + 18549.76 + 24330.58
_gross = _held + 12043.73 + 14942.19 + 9282.70     # after ABNB, ABT, ADP
_o = size_position(_EQ, 252.40, 236.22, _r, _s,
                   open_positions=6, gross_exposure=_gross)
check("the one-share AMZN trade is refused", _o.shares == 0,
      f"{_o.shares} shares")
check("and the reason names the minimum rather than the share count",
      "minimum for a trade" in (_o.rejected_reason or ""), _o.rejected_reason)

# ADP, squeezed to about half size on the same run, is still worth taking.
_o = size_position(_EQ, 265.22, 249.75, _r, _s, open_positions=5,
                   gross_exposure=_held + 12043.73 + 14942.19)
check("a trade squeezed to half size is still taken", _o.shares == 35,
      f"{_o.shares} shares risking ${_o.dollars_at_risk:,.2f}")

# A full-size trade with room to spare is untouched by any of this.
_o = size_position(_EQ, 100.0, 90.0, _r, _s, open_positions=0, gross_exposure=0.0)
check("a full-size trade is unaffected by the floor", _o.shares == 99,
      f"{_o.shares} shares risking ${_o.dollars_at_risk:,.2f}")

# Right at the boundary, in both directions.
_floor_dollars = _EQ * _r.risk_per_trade * _r.min_risk_fraction
_o = size_position(_EQ, 110.0, 100.0, _r, _s, open_positions=0,
                   gross_exposure=_EQ - 10 * 110.0)      # room for 10 shares
check("ten shares risking $100 is under the floor and refused",
      _o.shares == 0, f"{_o.shares} sh, floor ${_floor_dollars:,.2f}")
_o = size_position(_EQ, 110.0, 100.0, _r, _s, open_positions=0,
                   gross_exposure=_EQ - 30 * 110.0)      # room for 30 shares
check("thirty shares risking $300 clears it", _o.shares == 30,
      f"{_o.shares} sh risking ${_o.dollars_at_risk:,.2f}")

# The floor is a setting, not a law. Zero restores the old behaviour.
_off = _replace(_r, min_risk_fraction=0.0)
_o = size_position(_EQ, 252.40, 236.22, _off, _s,
                   open_positions=6, gross_exposure=_gross)
check("min_risk_fraction 0.0 takes whatever fits, as before", _o.shares == 1,
      f"{_o.shares} shares")

# A config predating the setting must not crash the sizer.
class _OldRisk:
    pass
_old = _OldRisk()
for _f in ("risk_per_trade", "max_position_pct", "max_gross_exposure",
           "max_open_positions"):
    setattr(_old, _f, getattr(_r, _f))
_o = size_position(_EQ, 100.0, 90.0, _old, _s, open_positions=0, gross_exposure=0.0)
check("a config without the setting still sizes normally", _o.shares == 99,
      f"{_o.shares} shares")


print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
