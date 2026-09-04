"""
End-to-end smoke test of the daily run.

The unit tests check pieces. This drives the whole `agent.py run` path with a
fake broker and made-up prices, so it catches the wiring mistakes that unit
tests miss: a function renamed but not re-imported, a variable used before it
is set, an exit path that never actually calls the broker.

It never touches the network and never touches a real account.

Run with:  python -m tests.smoke_run
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent as agent_mod
from tbot import data as datamod
from tbot import state as state_mod
from tbot.broker import Fill
from tests.synthetic import make_series

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if not cond and detail else ""))
    if not cond:
        FAILURES.append(name)


# --- a broker that records what it was told to do ---------------------------

class FakeBroker:
    def __init__(self, positions, orders, dry_run=False, **kw):
        self._positions = positions
        self._orders = orders
        self.dry_run = dry_run
        self.is_live = False
        self.submitted = []
        self.closed = []
        self.cancelled = []
        self.protected = []

    def account(self):
        return {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 200_000.0,
                "status": "ACTIVE", "pattern_day_trader": False,
                "trading_blocked": False, "mode": "PAPER"}

    def positions(self):
        return list(self._positions)

    def open_orders(self):
        return list(self._orders)

    def clock(self):
        return {"is_open": False}

    def submit_stop(self, symbol, shares, stop, allow_live=False, acknowledged=False):
        if self.dry_run:
            return Fill(symbol, shares, "", "dry-run", False, f"would protect {symbol}")
        self.protected.append({"symbol": symbol, "shares": shares, "stop": stop})
        self._orders.append({"symbol": symbol, "side": "sell", "id": "stop-" + symbol,
                             "qty": str(shares), "stop_price": str(stop)})
        return Fill(symbol, shares, "id", "accepted", True)

    def entry_dates(self):
        return {p["symbol"]: "2020-01-02" for p in self._positions}

    def realized_trades(self, limit=20):
        return []

    # Both of these mirror the real adapter: in dry-run mode they report what
    # they would have done and return submitted=False, so nothing downstream
    # records a trade that never happened.

    def close_position(self, symbol, allow_live=False, acknowledged=False):
        if self.dry_run:
            return Fill(symbol, 0, "", "dry-run", False, f"would close {symbol}")
        self.closed.append(symbol)
        self._positions = [p for p in self._positions if p["symbol"] != symbol]
        return Fill(symbol, 0, "id", "accepted", True)

    def submit_bracket(self, symbol, shares, stop, target, allow_live=False,
                       acknowledged=False):
        if self.dry_run:
            return Fill(symbol, shares, "", "dry-run", False,
                        f"would buy {shares} {symbol}")
        self.submitted.append({"symbol": symbol, "shares": shares,
                               "stop": stop, "target": target})
        return Fill(symbol, shares, "id", "accepted", True)


def force_signal(df):
    """Bend the last two bars until the entry rules actually fire.

    Random data almost never lands on a valid setup, so a smoke test built on
    it silently exercises nothing. This searches for a dip-then-reclaim that
    passes every condition, and raises if it cannot find one, so the test can
    never quietly pass by testing nothing.
    """
    from tbot.config import AgentConfig
    from tbot.strategy import latest_signal, prepare

    cfg = AgentConfig().strategy

    for dip in (0.97, 0.98, 0.985, 0.99):
        for pop in (1.02, 1.03, 1.04, 1.05):
            out = df.copy()
            ema_before = prepare(out.iloc[:-2], cfg)["ema_pb"].iloc[-1]

            i2, i1 = len(out) - 2, len(out) - 1
            c2 = float(ema_before) * dip
            out.iloc[i2, out.columns.get_loc("close")] = c2
            out.iloc[i2, out.columns.get_loc("open")] = c2 * 1.004
            out.iloc[i2, out.columns.get_loc("high")] = c2 * 1.008
            out.iloc[i2, out.columns.get_loc("low")] = c2 * 0.992

            ema_after = prepare(out.iloc[:-1], cfg)["ema_pb"].iloc[-1]
            c1 = float(ema_after) * pop
            out.iloc[i1, out.columns.get_loc("close")] = c1
            out.iloc[i1, out.columns.get_loc("open")] = c1 * 0.997
            out.iloc[i1, out.columns.get_loc("high")] = c1 * 1.003
            out.iloc[i1, out.columns.get_loc("low")] = c1 * 0.995

            if latest_signal("X", out, cfg) is not None:
                return out

    raise AssertionError("could not construct a valid entry setup")


def _end_today(df):
    """Slide the dates so the last bar is today.

    Without this the watchers correctly refuse to trade on data from 2017,
    which is the right behavior and makes the rest of the test meaningless.
    """
    idx = pd.bdate_range(end=pd.Timestamp.now("UTC").tz_localize(None).normalize(),
                         periods=len(df))
    out = df.copy()
    out.index = idx
    out.index.name = "date"
    return out


def build_bars():
    """UP is a live setup. TWIN is the same bet at a different price.
    DOWNTREND has rolled below its 50-day average. FLAT goes nowhere."""
    base = _end_today(make_series(n=500, seed=11, drift=0.0012, vol=0.008))
    up = force_signal(base)

    down = base.copy()
    tail = 60
    fade = np.linspace(1.0, 0.62, tail)
    for col in ("open", "high", "low", "close"):
        down.iloc[-tail:, down.columns.get_loc(col)] = (
            down[col].iloc[-tail:].to_numpy() * fade)

    twin = up.copy()
    for col in ("open", "high", "low", "close"):
        twin.iloc[:, twin.columns.get_loc(col)] = twin[col].to_numpy() * 1.37

    return {"UP": up, "DOWNTREND": down, "TWIN": twin,
            "FLAT": _end_today(make_series(n=500, seed=12, drift=0.0, vol=0.006))}


def run(positions, orders, submit=True, broker=None):
    bars = build_bars()
    if broker is None:
        broker = FakeBroker(positions, orders, dry_run=not submit)

    agent_mod.AlpacaBroker = lambda **kw: broker
    datamod.load_universe = lambda syms, start=None, end=None, refresh=False: bars

    tmp = Path(tempfile.mkdtemp())
    state_mod.STATE_DIR = tmp
    state_mod.STATE_FILE = tmp / "agent_state.json"
    state_mod.EQUITY_FILE = tmp / "equity_history.csv"
    state_mod.RUNLOG_FILE = tmp / "run_log.jsonl"

    # Send the generated page somewhere disposable. Without this the test
    # overwrites the real site/index.html with invented numbers, and the next
    # commit publishes fake positions to the live dashboard.
    from tbot import dashboard as dash_mod
    dash_mod.SITE = tmp
    agent_mod.write_dashboard = lambda title="Trading agent": (
        (tmp / "index.html").write_text(dash_mod.build_html(title=title))
        or (tmp / "index.html"))

    args = argparse.Namespace(
        symbols="UP,DOWNTREND,TWIN,FLAT", start="2016-01-01", equity=None,
        risk=None, refresh=True, submit=submit, no_research=True,
        i_understand_the_risk=False)

    agent_mod.cmd_run(args)
    return broker, state_mod.load_state()


# ---------------------------------------------------------------------------
print("\nA. A broken-down position gets sold")
# ---------------------------------------------------------------------------

b, st = run(
    positions=[{"symbol": "DOWNTREND", "shares": 10, "avg_entry": 100.0,
                "market_value": 700.0, "unrealized_pl": -300.0}],
    orders=[])

check("the position that lost its trend was closed", "DOWNTREND" in b.closed,
      str(b.closed))
check("the watcher noticed that position had no stop behind it",
      any("UNPROTECTED" in f["message"] for f in st["findings"]),
      str(st["findings"]))
check("an unprotected position stops new buying that run",
      st["orders"] == [], str(st["orders"]))
check("the exit was recorded with a reason",
      len(st["exits"]) == 1 and "trend break" in st["exits"][0]["reason"],
      str(st["exits"]))
check("a healthy position would not have been closed", b.closed == ["DOWNTREND"],
      str(b.closed))


# ---------------------------------------------------------------------------
print("\nB. A symbol with an unfilled order is not bought again")
# ---------------------------------------------------------------------------

b, st = run(positions=[], orders=[{"symbol": "UP", "id": "o1", "status": "accepted"}])
check("no second order was sent for the symbol already working",
      "UP" not in [o["symbol"] for o in b.submitted],
      str([o["symbol"] for o in b.submitted]))


# ---------------------------------------------------------------------------
print("\nC. Correlated duplicates are refused")
# ---------------------------------------------------------------------------

b, st = run(positions=[], orders=[])
bought = [o["symbol"] for o in b.submitted]
check("the setup actually fired, so this test is testing something",
      len(bought) >= 1, f"bought {bought}")
check("it did not buy both halves of the same bet",
      not ("UP" in bought and "TWIN" in bought), str(bought))
check("the duplicate was refused and the reason recorded",
      any("correlation" in s["reason"] for s in st["skipped"]),
      str(st["skipped"]))
check("the order carries a stop and a target",
      all(o["stop"] > 0 and o["target"] > o["stop"] for o in b.submitted),
      str(b.submitted))


# ---------------------------------------------------------------------------
print("\nD. Dry run sends nothing")
# ---------------------------------------------------------------------------

b, st = run(positions=[{"symbol": "DOWNTREND", "shares": 10, "avg_entry": 100.0,
                        "market_value": 700.0, "unrealized_pl": -300.0}],
            orders=[], submit=False)
check("no orders recorded in dry run", st["orders"] == [], str(st["orders"]))
check("nothing reached the broker in dry run",
      b.submitted == [] and b.closed == [],
      f"submitted={b.submitted} closed={b.closed}")
check("but it still worked out what it would have done",
      len(st["exits"]) == 1, str(st["exits"]))


# ---------------------------------------------------------------------------
print("\nE. Every run leaves a complete record")
# ---------------------------------------------------------------------------

b, st = run(positions=[], orders=[])
for key in ("mode", "account", "positions", "signals", "orders", "exits",
            "skipped", "vetoes", "updated_at", "healthy"):
    check(f"state records '{key}'", key in st)
check("mode is paper", st["mode"] == "paper", st["mode"])
check("no errors were logged", st["errors"] == [], str(st["errors"]))
check("equity history got a row", len(state_mod.load_equity_history()) == 1)

check("state records 'findings'", "findings" in st)
check("a clean run produces no critical findings",
      not [f for f in st["findings"] if f["severity"] == "critical"],
      str(st["findings"]))

from tbot import dashboard
html = dashboard.build_html(state=st, history=state_mod.load_equity_history())
check("the dashboard renders from that record", "<html" in html and "PAPER" in html)

# ---------------------------------------------------------------------------
print("\nE1b. An unprotected position gets a stop, it is not just reported")
# ---------------------------------------------------------------------------
# Writing "UNPROTECTED" into a log and moving on leaves the position exactly
# as exposed as it was. Nothing else in the run covers it: entries only open
# new trades and the soft exits wait for a close that may be days away.

_naked_pos = [{"symbol": "UP", "shares": 10, "avg_entry": 100.0,
               "market_value": 1_000.0, "unrealized_pl": 0.0}]
b, st = run(positions=_naked_pos, orders=[])
check("a stop was actually placed, not merely noted",
      [p["symbol"] for p in b.protected] == ["UP"], str(b.protected))
check("the stop covers every share held",
      b.protected and b.protected[0]["shares"] == 10, str(b.protected))
check("the stop sits below the market, so it cannot fire on submission",
      b.protected and b.protected[0]["stop"] < 100.0 * 5, str(b.protected))
check("the repair is recorded in state", len(st.get("protected") or []) == 1,
      str(st.get("protected")))
check("the watcher still reported the problem it fixed",
      any("UNPROTECTED" in f["message"] for f in st["findings"]))

# A position that already has a full stop must not collect a second one.
_covered = run(
    positions=[{"symbol": "UP", "shares": 10, "avg_entry": 100.0,
                "market_value": 1_000.0, "unrealized_pl": 0.0}],
    orders=[{"symbol": "UP", "side": "sell", "id": "s1", "qty": "10",
             "stop_price": "90"}])[0]
check("a position that is already protected is left alone",
      _covered.protected == [], str(_covered.protected))

# Half a stop is still an exposure, and only the uncovered shares need cover.
_half = run(
    positions=[{"symbol": "UP", "shares": 10, "avg_entry": 100.0,
                "market_value": 1_000.0, "unrealized_pl": 0.0}],
    orders=[{"symbol": "UP", "side": "sell", "id": "s1", "qty": "4",
             "stop_price": "90"}])[0]
check("a partly covered position gets a stop for the remaining shares only",
      _half.protected and _half.protected[0]["shares"] == 6, str(_half.protected))


# ---------------------------------------------------------------------------
print("\nE2. One broker failure does not take the whole run down")
# ---------------------------------------------------------------------------
# A run that dies partway through is worse than a run that does nothing: state
# is never saved, the dashboard still shows yesterday as current, and the
# remaining positions go unmanaged with no error anywhere a person would see.

from tbot.broker import BrokerError as _BE


class RejectingBroker(FakeBroker):
    """The broker refuses every buy, the way it would on a halted symbol or
    with the buying power already spent."""
    def submit_bracket(self, symbol, shares, stop, target, allow_live=False,
                       acknowledged=False):
        raise _BE("POST /v2/orders -> 403: insufficient buying power")


b, st = run(positions=[], orders=[], broker=RejectingBroker([], []))
check("a rejected order is recorded rather than crashing the run",
      any("rejected" in e for e in st["errors"]), str(st["errors"]))
check("the run still finished and saved its state", bool(st.get("updated_at")))
check("the rejected symbol is listed as skipped, with the reason",
      any("broker rejected" in x.get("reason", "") for x in st["skipped"]),
      str(st["skipped"]))
check("nothing is recorded as an order that was never accepted",
      st["orders"] == [], str(st["orders"]))


class FailingExitBroker(FakeBroker):
    """The close fails, which is the moment a position is least protected."""
    def close_position(self, symbol, allow_live=False, acknowledged=False):
        raise _BE(f"could not close {symbol}: 422. WARNING: {symbol} is now "
                  f"held with no stop order behind it.")


_pos = [{"symbol": "DOWNTREND", "shares": 10, "avg_entry": 100.0,
         "market_value": 700.0, "unrealized_pl": -300.0}]
b, st = run(positions=_pos, orders=[], broker=FailingExitBroker(list(_pos), []))
check("a failed exit is recorded rather than crashing the run",
      any("could not close" in e for e in st["errors"]), str(st["errors"]))
check("the failed exit run still saved its state", bool(st.get("updated_at")))
check("the run is marked unhealthy after a failed exit", st["healthy"] is False)


# ---------------------------------------------------------------------------
print("\nE3. An unfinished bar is not tradeable")
# ---------------------------------------------------------------------------
# Every decision reads the most recent daily close. During the session that
# close has not happened yet, so a mid-day run would size entries and trigger
# exits off a number that is still moving.

class OpenMarketBroker(FakeBroker):
    def clock(self):
        return {"is_open": True}


b, st = run(positions=[], orders=[], broker=OpenMarketBroker([], []))
check("no orders are sent while the market is still open",
      b.submitted == [], str(b.submitted))
check("nothing is closed on an unfinished bar either", b.closed == [], str(b.closed))
check("the mid-session run says why it stopped",
      any("market hours" in e for e in st["errors"]), str(st["errors"]))


# ---------------------------------------------------------------------------
print("\nF. The watchers refuse to trade on a broken picture")
# ---------------------------------------------------------------------------

from tbot import watch

_stale = build_bars()
for k in _stale:
    _stale[k].index = _stale[k].index - pd.Timedelta(days=45)
_f = watch.check_data(_stale, list(_stale))
check("stale prices are a critical finding",
      any(x.severity == watch.CRITICAL and "stale" in x.message for x in _f),
      str([x.message for x in _f]))

_frozen = build_bars()
_col = _frozen["FLAT"].columns.get_loc("close")
_frozen["FLAT"].iloc[-6:, _col] = 100.0
check("a frozen price feed is caught",
      any("frozen" in x.message for x in watch.check_data(_frozen, list(_frozen))))

check("a missing symbol is reported",
      any("failed to load" in x.message
          for x in watch.check_data(build_bars(), list(build_bars()) + ["GONE"])))

_acct = {"equity": 100_000.0, "buying_power": 200_000.0, "trading_blocked": False}
_naked = watch.check_broker(_acct, [{"symbol": "AAA", "market_value": 5_000}], [])
check("a position with no sell order is critical",
      any(x.severity == watch.CRITICAL and "UNPROTECTED" in x.message for x in _naked))

_ok = watch.check_broker(
    _acct, [{"symbol": "AAA", "shares": 10, "market_value": 5_000}],
    [{"symbol": "AAA", "side": "sell", "id": "1", "qty": "10", "stop_price": "90"}])
check("a properly protected position raises nothing critical",
      not [x for x in _ok if x.severity == watch.CRITICAL], str(_ok))

# A bracket has two sell legs and only one of them is an exit. Counting sell
# orders instead of reading them would call this position protected.
_limit_only = watch.check_broker(
    _acct, [{"symbol": "AAA", "shares": 10, "market_value": 5_000}],
    [{"symbol": "AAA", "side": "sell", "id": "1", "qty": "10", "limit_price": "120"}])
check("a take-profit limit alone is not treated as protection",
      any(x.severity == watch.CRITICAL and "UNPROTECTED" in x.message
          for x in _limit_only), str(_limit_only))

_partial = watch.check_broker(
    _acct, [{"symbol": "AAA", "shares": 10, "market_value": 5_000}],
    [{"symbol": "AAA", "side": "sell", "id": "1", "qty": "4", "stop_price": "90"}])
check("a stop covering only part of the position is critical",
      any(x.severity == watch.CRITICAL and "PARTLY UNPROTECTED" in x.message
          for x in _partial), str(_partial))

_nested = watch.check_broker(
    _acct, [{"symbol": "AAA", "shares": 10, "market_value": 5_000}],
    [{"symbol": "AAA", "side": "buy", "id": "p", "legs": [
        {"symbol": "AAA", "side": "sell", "id": "l1", "qty": "10", "stop_price": "90"}]}])
check("a stop nested under its parent order still counts",
      not [x for x in _nested if x.severity == watch.CRITICAL], str(_nested))

check("a held position with no price data is critical",
      any(x.severity == watch.CRITICAL for x in
          watch.check_positions_have_data([{"symbol": "GONE"}], build_bars())))
check("a held position that does have data is fine",
      not watch.check_positions_have_data(
          [{"symbol": sorted(build_bars())[0]}], build_bars()))

check("blocked trading at the broker is critical",
      any(x.severity == watch.CRITICAL for x in
          watch.check_broker({**_acct, "trading_blocked": True}, [], [])))
check("an oversized position is flagged even though no rule created it",
      any("of the account" in x.message for x in
          watch.check_broker(_acct, [{"symbol": "AAA", "market_value": 50_000}],
                             [{"symbol": "AAA", "side": "sell", "id": "1",
                               "qty": "10", "stop_price": "90"}])))

_old = {"updated_at": "2026-01-01T00:00:00+00:00", "positions": []}
check("a schedule that stopped firing is critical",
      any(x.severity == watch.CRITICAL and "schedule" in x.message
          for x in watch.check_run_health(_old, [])))

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} SMOKE CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("Smoke test passed: the full daily run works end to end.")
