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
cols = ["sma_fast", "sma_slow", "ema_pb", "atr", "swing_low", "adv"]
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
                   cfg_strategy=cfg.strategy, open_positions=5)
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

datamod.CACHE_DIR = _cache_backup


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
check("momentum reads a trailing return, not today's bar",
      ranking.score("momentum", _rk_df, _last, _stop) ==
      ranking.score("momentum", _rk_df.copy(), _last, _stop))

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
print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
