#!/usr/bin/env python3
"""
Trend pullback trading agent.

  python agent.py selftest              check the engine's math
  python agent.py backtest              test the rules on history
  python agent.py scan                  today's signals, no orders
  python agent.py paper                 send today's signals to a paper account
  python agent.py status                paper account and open positions
  python agent.py cache                 what market data is stored locally

Nothing sends a real order without three separate switches being thrown.
See broker.py.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from tbot import data as datamod
from tbot import report as rep
from tbot import state
from tbot.backtest import run_backtest
from tbot.broker import (AlpacaBroker, BrokerError, committed_symbols,
                         describe_safety)
from tbot.config import DEFAULT_WATCHLIST, AgentConfig
from tbot.dashboard import write_dashboard
from tbot.research import Researcher, screen
from tbot.risk import DrawdownMonitor, correlation_block, size_position
from tbot import ranking
from tbot import watch
from tbot.strategy import exit_decision, latest_signal, prepare, trend_state


def load_dotenv():
    """Read a .env file next to this script, if one exists."""
    env = Path(__file__).resolve().parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_config(args) -> AgentConfig:
    cfg = AgentConfig()
    if getattr(args, "equity", None):
        cfg.risk.starting_equity = args.equity
    if getattr(args, "risk", None):
        cfg.risk.risk_per_trade = args.risk / 100.0
    if getattr(args, "symbols", None):
        cfg.watchlist = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    return cfg


# ---------------------------------------------------------------------------

def cmd_backtest(args):
    cfg = build_config(args)
    print(f"Loading {len(cfg.watchlist)} symbols from {args.start}...")
    bars = datamod.load_universe(cfg.watchlist, start=args.start, end=args.end,
                                 refresh=args.refresh)
    print(f"Loaded {len(bars)} symbols. Running backtest...\n")

    result = run_backtest(bars, cfg)
    s = result.stats()

    if s.get("trades", 0) == 0:
        print("No trades were generated. Check your date range and watchlist.")
        return

    print(f"  Period            {result.equity.index[0].date()} to {result.equity.index[-1].date()}")
    print(f"  Starting equity   ${s['start_equity']:,.2f}")
    print(f"  Ending equity     ${s['end_equity']:,.2f}")
    print(f"  Total return      {s['total_return_pct']:+.2f}%")
    print(f"  CAGR              {s['cagr_pct']:+.2f}%")
    print(f"  Max drawdown      -{s['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe            {s['sharpe']:.2f}")
    print()
    print(f"  Trades            {s['trades']}")
    print(f"  Win rate          {s['win_rate_pct']:.1f}%")
    print(f"  Avg win           ${s['avg_win']:,.2f}")
    print(f"  Avg loss          ${s['avg_loss']:,.2f}")
    print(f"  Expectancy        {s['expectancy_R']:+.3f}R per trade")
    print(f"  Profit factor     {s['profit_factor']:.2f}")
    print(f"  Avg hold          {s['avg_days_held']:.1f} days")
    print(f"  Time in market    {s['exposure_pct']:.1f}%")
    print(f"  Exits             {s['exit_breakdown']}")

    path = rep.write_backtest_html(result)
    csv = rep.REPORTS / "trades.csv"
    result.trades_df().to_csv(csv, index=False)
    print(f"\n  Report  {path}")
    print(f"  Trades  {csv}")

    # Buy and hold on SPY over the same window, for context. A strategy that
    # underperforms doing nothing is not a strategy.
    if "SPY" in bars:
        spy = bars["SPY"]
        spy = spy[(spy.index >= result.equity.index[0]) & (spy.index <= result.equity.index[-1])]
        if len(spy) > 1:
            bh = (spy["close"].iloc[-1] / spy["close"].iloc[0] - 1) * 100
            print(f"\n  For comparison, buying and holding SPY over the same period: {bh:+.2f}%")


def cmd_scan(args):
    cfg = build_config(args)
    equity = cfg.risk.starting_equity

    if args.use_broker:
        load_dotenv()
        try:
            equity = AlpacaBroker().account()["equity"]
            print(f"Using live account equity: ${equity:,.2f}")
        except BrokerError as exc:
            print(f"Could not read account equity ({exc}). Falling back to config.")

    print(f"Scanning {len(cfg.watchlist)} symbols...")
    bars = datamod.load_universe(cfg.watchlist, start=args.start, refresh=args.refresh)

    cards, rejected, as_of = [], [], None
    for sym, df in bars.items():
        as_of = df.index[-1] if as_of is None else max(as_of, df.index[-1])
        sig = latest_signal(sym, df, cfg.strategy)
        if sig is None:
            continue
        order = size_position(equity, sig.reference_close, sig.stop,
                              cfg.risk, cfg.strategy)
        if not order.ok:
            rejected.append(f"{sym}: {order.rejected_reason}")
            continue
        cards.append(rep.trade_card(sym, order, sig, trend_state(df, cfg.strategy)))

    text = rep.signal_report(cards, rejected, equity, len(bars), as_of, cfg)
    path = rep.write_signal_report(text, as_of)
    print("\n" + text)
    print(f"\nSaved to {path}")

    stale = pd.Timestamp.now().normalize() - pd.Timestamp(as_of).normalize()
    if stale.days > 4:
        print(f"\nWARNING: newest bar is {stale.days} days old. Run with --refresh.")


# The in-progress record, so a crash can still publish what it knows.
_LIVE_STATE = {}


def cmd_run(args):
    """The daily job, wrapped so that a crash still leaves a record.

    Without this an unhandled exception, from a data source changing its
    schema, a symbol with a missing column, a typo in the config, ends the run
    in a traceback before anything is saved. state/agent_state.json still holds
    yesterday's snapshot and the dashboard renders it as today's, so a broken
    run is indistinguishable from a quiet one for three days, which is how long
    the watchdog takes to notice the schedule stopped producing.
    """
    try:
        return _cmd_run(args)
    except BaseException as exc:
        st = _LIVE_STATE
        if st:
            st.setdefault("errors", []).append(
                f"the run failed partway through: {type(exc).__name__}: {exc}")
            st["healthy"] = False
            try:
                # Same reason as the early returns. save_state replaces the file
                # wholesale, so a crash before the account was read publishes
                # blank_state's defaults -- $0.00 equity, no positions -- and the
                # page reads like a wiped-out account rather than a failed run.
                # The file on disk still holds the last completed run.
                if not st.get("account"):
                    _carry_display(st, state.load_state(),
                                   f"the run failed partway through "
                                   f"({type(exc).__name__})")
                state.save_state(st)
                write_dashboard()
            except Exception:
                pass   # already failing; do not fail differently
        print(f"\nRUN FAILED: {type(exc).__name__}: {exc}")
        print("The record and the dashboard were written so the failure is visible.")
        raise


def cmd_monitor(args):
    """A mid-session safety check. Reads the account, runs the watchers, puts a
    stop behind anything holding none, and rebuilds the dashboard. It opens
    nothing and closes nothing.

    Why it may not trade: every entry, stop and exit in this system reads the
    most recent daily bar. While the market is open that bar is still moving,
    so a mid-session decision would be made on a close that has not happened.
    The backtest only ever saw finished bars; letting the live agent act on
    unfinished ones would make the two stop describing the same strategy, and
    the backtest would stop predicting anything about the live account.

    A missing stop is the exception, and it is why this exists. Waiting until
    after the close to notice that a position has nothing behind it leaves it
    naked for a whole session. The stop level is a documented approximation
    either way, so placing it now on a moving price loses very little and
    removes hours of exposure.

    It deliberately writes no equity row and no run-log entry. Those feed the
    drawdown breaker and the schedule watchdog, which count one trading run per
    day; a second daily row would tell both of them a story that is not true.
    """
    load_dotenv()
    cfg = build_config(args)

    broker = AlpacaBroker(dry_run=not args.submit)

    # Carry the previous run's record forward. This run refreshes the account
    # and the checks, but it did not scan, so it has no signals or orders of
    # its own. Writing blanks over the day's record would erase the trading
    # run's work from the page every single midday.
    prev = state.load_state() or {}
    st = state.blank_state()
    for key, value in prev.items():
        if key in st:
            st[key] = value

    st["mode"] = "live" if broker.is_live else "paper"
    # The dashboard says so out loud. Without it the page carries last night's
    # orders under a timestamp from lunchtime, which reads as though the agent
    # had just placed them.
    st["monitor_only"] = True
    st["errors"] = []
    st["protected"] = []
    # The loop above copies every key blank_state defines, which includes the
    # carry markers an aborted run may have left behind. This check does read
    # the account, so leaving them set would have the page warn that the
    # figures are old at the moment they were re-measured.
    st["carried_from"] = None
    st["carried_reason"] = ""

    print(describe_safety(broker, cfg))
    if broker.is_live and not (cfg.allow_live_trading and args.i_understand_the_risk):
        print("Refusing to continue against a live endpoint. Exiting.")
        return

    try:
        acct = broker.account()
        positions = broker.positions()
    except BrokerError as exc:
        print(f"\nCannot reach the broker: {exc}")
        st["errors"].append(f"broker unreachable: {exc}")
        st["healthy"] = False
        _carry_display(st, prev, "the broker could not be reached")
        state.save_state(st)
        write_dashboard()
        return

    try:
        orders_open = broker.open_orders()
    except BrokerError as exc:
        # Unknown is not empty. Every held position would read as unprotected,
        # and this run would stack a second stop behind stops that already
        # exist. One of them filling then leaves a naked short.
        print(f"Cannot read working orders: {exc}")
        print("Refusing to act on an unknown order book.")
        st["errors"].append(f"could not read working orders: {exc}")
        st["account"] = acct
        st["positions"] = positions
        st["healthy"] = False
        state.save_state(st)
        write_dashboard()
        return

    st["account"] = acct
    st["positions"] = positions
    equity = acct["equity"]
    print(f"Account: {acct['mode']}  equity ${equity:,.2f}")
    print(f"Holding {len(positions)}: "
          f"{sorted(p['symbol'] for p in positions) or 'nothing'}\n")

    # Only what is held. The watchlist is not needed to answer "is everything I
    # own protected", and downloading all of it would make a check that should
    # take seconds take minutes.
    syms = sorted({p["symbol"] for p in positions})
    bars = datamod.load_universe(syms, start=args.start, refresh=True) if syms else {}

    prior_equity = state.load_equity_history()
    prior_runs = state.load_runs()

    def look(orders):
        found = watch.run_all(bars, syms, acct, positions, orders,
                              prev or None, prior_equity, runs=prior_runs)
        if found:
            print(f"Checks: {watch.summarize(found)}")
            for f in found:
                if f.severity != watch.INFO:
                    print(f"  [{f.severity.upper()}] {f.agent}: {f.message}")
            print()
        return found

    findings = look(orders_open)

    repaired = protect_exposed(broker, cfg, bars, positions, orders_open, st,
                               acknowledged=args.i_understand_the_risk)

    # The watchers judged a picture this run has since changed.
    if repaired:
        try:
            orders_open = broker.open_orders()
        except BrokerError as exc:
            st["errors"].append(f"could not re-read orders after protecting: {exc}")
        else:
            print(f"Re-checking after protecting {', '.join(repaired)}.")
            findings = look(orders_open)

    st["findings"] = [f.to_dict() for f in findings]
    st["healthy"] = not st["errors"]
    # save_state stamps updated_at itself.

    state.save_state(st)
    write_dashboard()
    print("Monitoring check complete. No entries or exits were considered.")


def protect_exposed(broker, cfg, bars, positions, orders_open, st,
                    acknowledged=False):
    """Put a stop behind every held position that has none. Returns the list of
    symbols actually protected.

    Detecting an unprotected position and then only writing it down leaves it
    unprotected. Nothing else fixes it either: the entry path only opens new
    trades, and the exit path waits for a close below an average that may be
    days away. So the repair happens here, first, ahead of the decision to stop
    trading on a critical finding, because the position is exposed for exactly
    as long as nobody acts.

    The level is two ATRs below the last close. That is the same distance the
    breakout rules use, it cannot sit above the market and fire instantly, and
    it is wide enough to survive ordinary noise. It is a guess, but a
    documented one, and a guessed stop beats no stop.

    Shared by the daily run and the mid-session monitor. Two copies of this
    would drift, and the copy that drifted would be the one holding the only
    safety net under a position nobody is watching.
    """
    exposed = watch.unprotected(positions, orders_open)
    # Shares reserved by an orphaned take-profit leg. Its bracket lost the stop
    # and kept the target, so the position is bare while every share is spoken
    # for, and a replacement stop comes back 403 insufficient qty. Cancelling
    # the survivor is the only way to get a stop in, and the replacement goes
    # back as an OCO pair so the target is not simply thrown away.
    blocked = watch.orphaned_targets(positions, orders_open)
    repaired = []
    for sym, shares in sorted(exposed.items()):
        df = bars.get(sym)
        if df is None or len(df) < cfg.strategy.atr_period + 2:
            st["errors"].append(f"cannot protect {sym}: no usable price data")
            continue
        prepared = prepare(df, cfg.strategy)
        last = prepared.iloc[-1]
        atr_val = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        close = float(last["close"])
        if atr_val <= 0 or close <= 0:
            st["errors"].append(f"cannot protect {sym}: no usable ATR")
            continue
        level = round(close - 2.0 * atr_val, 2)

        # The orphaned target's own price, so the replacement keeps the exit
        # the trade was opened with rather than inventing a new one.
        loose = blocked.get(sym) or []
        target = 0.0
        for o in loose:
            try:
                target = max(target, float(o.get("limit_price") or 0))
            except (TypeError, ValueError):
                continue

        if loose:
            try:
                gone = broker.cancel_order_ids([o.get("id") for o in loose])
            except BrokerError as exc:
                print(f"  {sym}: COULD NOT CLEAR THE ORPHANED TARGET, {exc}")
                st["errors"].append(
                    f"could not cancel the orphaned target on {sym}, so no stop "
                    f"could be placed: {exc}")
                continue
            if not gone and not broker.dry_run:
                st["errors"].append(
                    f"the orphaned target on {sym} could not be cancelled, so "
                    f"the shares stay reserved and unprotected")
                continue
            print(f"  {sym}: cancelled {len(gone)} orphaned target order(s) "
                  f"holding the shares")

        try:
            if loose and target > close:
                # Both legs, linked, replacing the pair the bracket lost.
                fill = broker.submit_protective_oco(
                    sym, shares, level, target,
                    allow_live=cfg.allow_live_trading,
                    acknowledged=acknowledged, last_price=close)
                if not fill.submitted and fill.status == "rejected":
                    # The pair was refused. A stop alone is worth more than a
                    # target alone, and the target is already cancelled.
                    print(f"  {sym}: OCO refused ({fill.detail}), "
                          f"falling back to a plain stop")
                    fill = broker.submit_stop(
                        sym, shares, level, allow_live=cfg.allow_live_trading,
                        acknowledged=acknowledged, last_price=close)
            else:
                fill = broker.submit_stop(sym, shares, level,
                                          allow_live=cfg.allow_live_trading,
                                          acknowledged=acknowledged,
                                          last_price=close)
        except BrokerError as exc:
            print(f"  {sym}: COULD NOT PROTECT, {exc}")
            if loose:
                # Worth saying plainly: the blocking order was removed and
                # nothing replaced it, so the position is now bare of orders
                # entirely rather than bare of protection.
                st["errors"].append(
                    f"could not place a stop on {sym} after cancelling its "
                    f"orphaned target, so it now has no orders at all: {exc}")
            else:
                st["errors"].append(f"could not place a stop on {sym}: {exc}")
            continue
        if not fill.submitted:
            # A dry run sends nothing. Printing PROTECTED and writing it into
            # the record would be a false entry in the one log you most need to
            # be able to trust.
            print(f"  {sym}: would protect {shares} sh at ${level:,.2f} "
                  f"({fill.status})")
            continue
        print(f"  {sym}: PROTECTED, stop on {shares} sh at ${level:,.2f} "
              f"({fill.status})")
        st["protected"].append({"symbol": sym, "shares": shares, "stop": level,
                                "target": target if loose and target > close else None,
                                "replaced_orphaned_target": bool(loose),
                                "status": fill.status})
        repaired.append(sym)
    return repaired


# What the dashboard draws. A run that stops early has none of it, and
# save_state replaces the file wholesale rather than merging into it, so
# without carrying these forward an aborted run publishes blank_state's
# defaults as though they were measurements: $0.00 equity, no positions, no
# briefing. Nothing about the account changed -- only the page did, and it read
# like the account had been wiped out.
_DISPLAY_KEYS = ("account", "positions", "signals", "vetoes", "orders", "exits",
                 "protected", "skipped", "findings", "briefing", "recent_trades")


def _carry_display(st, prev, reason):
    """Copy the last good run's display block onto a run that stopped early.

    The figures stay a single consistent snapshot from one run rather than a
    mixture of fresh and stale, and `carried_from` dates them by the run they
    actually came from so the page can say so out loud. A first-ever run has
    nothing to carry, and blank is then the truth.
    """
    if not (prev or {}).get("updated_at"):
        return st
    for key in _DISPLAY_KEYS:
        if key in prev:
            st[key] = copy.deepcopy(prev[key])
    st["carried_from"] = prev["updated_at"]
    st["carried_reason"] = reason
    return st


def _cmd_run(args):
    """The daily job: rules find setups, research screens them, orders go in,
    everything is recorded, the dashboard is rebuilt."""
    load_dotenv()
    cfg = build_config(args)
    if args.no_research:
        cfg.research.use_llm = False
        cfg.research.check_earnings = False

    broker = AlpacaBroker(dry_run=not args.submit)
    st = state.blank_state()
    _LIVE_STATE.clear()
    _LIVE_STATE.update(st)
    st = _LIVE_STATE
    st["mode"] = "live" if broker.is_live else "paper"

    # Carry forward what the previous run knew before anything can return
    # early. save_state replaces the file wholesale, so without this a run that
    # stops at the market-clock guard writes a blank map, and the next run has
    # no record of which strategy opened which position. A mean reversion trade
    # would then be managed by the trend exit and closed the day after it was
    # opened, at a loss, for no reason anybody could see.
    _prev = state.load_state()
    st["strategy_by_symbol"] = dict((_prev or {}).get("strategy_by_symbol") or {})

    print(describe_safety(broker, cfg))
    if broker.is_live and not (cfg.allow_live_trading and args.i_understand_the_risk):
        print("Refusing to continue against a live endpoint. Exiting.")
        return

    try:
        acct = broker.account()
        positions = broker.positions()
    except BrokerError as exc:
        print(f"\nCannot reach the broker: {exc}")
        print("Set ALPACA_API_KEY and ALPACA_API_SECRET in .env (see .env.example).")
        st["errors"] = [f"broker unreachable: {exc}"]
        _carry_display(st, _prev, "the broker could not be reached")
        state.save_state(st)
        # Rebuild the page even on failure, so the dashboard reports the
        # outage instead of silently showing yesterday's numbers as current.
        write_dashboard()
        return

    equity, cash = acct["equity"], acct["cash"]

    # Every decision below reads the most recent daily bar. While the market is
    # open that bar is half a day old and still moving, so a close that has not
    # happened yet would set the trend filter, the entries and the exits. The
    # backtest never sees a partial bar and neither should this. A run during
    # the session reports and stops.
    try:
        market_open = bool(broker.clock().get("is_open"))
    except (BrokerError, AttributeError) as exc:
        # The account and the positions were read successfully seconds ago, so
        # a failure here is the clock endpoint specifically. Assuming "closed"
        # let the run set entries, stops, targets and every trend-break exit
        # from a daily bar that is still moving. Unknown is not closed, and a
        # missed day costs nothing but opportunity.
        st["errors"].append(f"could not read the market clock: {exc}")
        market_open = True
    if market_open and not getattr(args, "ignore_session", False):
        print("The market is open, so today's bar is not finished yet.\n"
              "This agent trades on completed daily bars. Run it after the "
              "close, or pass --ignore-session to override.")
        st["errors"].append("run attempted during market hours")
        # Record what the broker already told us. Refusing to trade is not the
        # same as having nothing: without these two lines the saved state keeps
        # blank_state's empty account and position list, and the dashboard
        # redraws as though the account held nothing at all. On a first-ever
        # run this is all there is; after that _carry_display below replaces
        # them with the last complete snapshot, so the page shows one coherent
        # set of figures under one timestamp instead of a mixture.
        st["account"] = acct
        st["positions"] = positions
        _carry_display(st, _prev, "the run was started while the market was open")
        state.save_state(st)
        write_dashboard()
        return

    # Count orders that are accepted but not yet filled as already owned.
    # Without this, a second run before the market opens submits the same
    # trade again and silently doubles the risk on it.
    try:
        orders_open = broker.open_orders()
    except BrokerError as exc:
        # An empty list would mean "there are none". This means "I do not know",
        # and the two are not interchangeable. Treating unknown as none makes
        # every held position look unprotected, so the repair path stacks a
        # second stop behind stops that already exist, and one of them filling
        # leaves a naked short. It also makes pending buys stop counting as
        # commitments, so the same trade goes in twice.
        print(f"Cannot read working orders: {exc}")
        print("Refusing to act on an unknown order book.")
        st["errors"].append(f"could not read working orders: {exc}")
        st["account"] = acct
        st["positions"] = positions
        _carry_display(st, _prev, "the working orders could not be read")
        state.save_state(st)
        write_dashboard()
        return

    held, working = committed_symbols(positions, orders_open)
    pending_only = sorted(working - held)
    held |= working

    gross = sum(p["market_value"] for p in positions)
    open_count = len(held)

    st["account"] = acct
    st["positions"] = positions
    print(f"Account: {acct['mode']}  equity ${equity:,.2f}  "
          f"buying power ${acct['buying_power']:,.2f}")
    print(f"Committed to {open_count}: {sorted(held) or 'nothing'}")
    if pending_only:
        print(f"  (orders already working, not yet filled: {pending_only})")
    print()

    if acct.get("trading_blocked"):
        st["errors"].append("trading blocked on this account")
        st["account"] = acct
        st["positions"] = positions
        _carry_display(st, _prev, "trading is blocked on this account")
        state.save_state(st)
        write_dashboard()
        print("Trading is blocked on this account. Exiting.")
        return

    researcher = Researcher(enabled=cfg.research.use_llm)
    st["research"]["enabled"] = researcher.enabled
    if cfg.research.use_llm and not researcher.enabled:
        print("No ANTHROPIC_API_KEY found, so news review is off. "
              "The earnings filter still runs.\n")

    # Held positions have to be in here even when they are no longer on the
    # watchlist. Without their bars the exit loop below skips them silently
    # ("if df is None: continue"), protect_exposed cannot price a stop for them,
    # and check_positions_have_data raises a critical that empties the candidate
    # list -- so deleting one delisted name from the watchlist orphans a live
    # position AND stops the agent trading anything else. cmd_monitor already
    # loads what is held; this is the path that did not.
    bars = datamod.load_universe(
        sorted(set(cfg.watchlist) | {p["symbol"] for p in positions}),
        start=args.start, refresh=True)

    # --- the watchers, before any decision is made on this data -------------
    previous = _prev
    prior_equity = state.load_equity_history()
    prior_runs = state.load_runs()

    # The drawdown circuit breaker. It existed in config, in risk.py and in the
    # backtest, and was the one risk control the live run never consulted, so a
    # 30% drawdown would have kept opening full-size positions all the way down.
    # The first argument is the starting high water mark, not the limit. Passing
    # the limit there shifted every argument one slot left, so the breaker read
    # limit=0.10 and resume_below=min(60, 0.10)=0.10: it halted at a 10%
    # drawdown instead of the configured 20%, and the hysteresis that keeps it
    # from flip-flopping on the boundary was gone because the two thresholds had
    # collapsed onto each other. peak is max(peak, equity) on every update, so
    # seeding it at 0 and replaying the history below rebuilds the true peak.
    dd = DrawdownMonitor(0.0, cfg.risk.max_drawdown_halt,
                         cfg.risk.resume_below, cfg.risk.halt_cooldown_days)
    # append_equity replaces today's row rather than appending a second one, so
    # on a re-run today is already in this history. Replaying it and then
    # calling update(equity) again feeds the same day twice, which inflates
    # days_tripped and lets the cooldown expire early.
    _today = datetime.now(timezone.utc).date().isoformat()
    for row in prior_equity:
        if str(row.get("date", ""))[:10] == _today:
            continue
        try:
            dd.update(float(row["equity"]))
        except (KeyError, TypeError, ValueError):
            continue
    halted = dd.update(equity)
    if halted:
        st["errors"].append(
            f"drawdown halt: equity is {(1 - equity / dd.peak) * 100:.1f}% below "
            f"its high of ${dd.peak:,.2f}. No new positions until it recovers "
            f"to within {cfg.risk.resume_below * 100:.0f}%.")
        print(f"DRAWDOWN HALT: {(1 - equity / dd.peak) * 100:.1f}% below the "
              f"high water mark. Managing existing positions only.\n")

    def look(orders):
        found = watch.run_all(bars, cfg.watchlist, acct, positions, orders,
                              previous, prior_equity, runs=prior_runs)
        if found:
            print(f"Checks: {watch.summarize(found)}")
            for f in found:
                if f.severity != watch.INFO:
                    print(f"  [{f.severity.upper()}] {f.agent}: {f.message}")
            print()
        return found

    findings = look(orders_open)

    # --- put a stop behind anything that has none, before anything else -----
    repaired = protect_exposed(broker, cfg, bars, positions, orders_open, st,
                               acknowledged=args.i_understand_the_risk)

    # The watchers judged a picture this run has since changed. Re-reading the
    # broker is the only honest way to know whether the problem is still there:
    # otherwise the agent refuses to trade all day on the strength of a finding
    # it fixed itself two seconds earlier, and it would do that every run.
    if repaired:
        try:
            orders_open = broker.open_orders()
        except BrokerError as exc:
            st["errors"].append(f"could not re-read orders after protecting: {exc}")
        else:
            print(f"Re-checking after protecting {', '.join(repaired)}.")
            findings = look(orders_open)
            held, working = committed_symbols(positions, orders_open)
            held |= working
            # These are what the position limits are judged against, so they
            # have to move with it. Refreshing `held` alone let a commitment
            # accepted between the two reads slip past the max-positions cap.
            open_count = len(held)

    st["findings"] = [f.to_dict() for f in findings]
    criticals = [f for f in findings if f.severity == watch.CRITICAL]

    # A critical finding means the agent's picture of the world is wrong.
    # Placing new orders on a wrong picture is how a small fault becomes an
    # expensive one, so it manages what it already holds and stops there.
    if criticals:
        st["errors"] += [f.message for f in criticals]
        print("Critical checks failed. No new positions will be opened this run.\n")

    # --- manage what is already open, before looking for anything new -------
    # The stop and target are live at the broker and need no help. These two
    # exits depend on the daily close, so nothing but this run can act on them.
    # Which strategy opened each position, carried forward from the last run.
    # Anything unknown is managed by the pullback exits, which is the oldest
    # and most conservative set, and is noted rather than assumed silently.
    # An alias, not a copy. st["strategy_by_symbol"] is already a fresh dict
    # built at the top of this run, and the crash handler saves _LIVE_STATE --
    # which is this same st. As a copy, a run that submitted a reversion order
    # and then died anywhere before the prune at the end lost the record of
    # which strategy opened it, and tomorrow the pullback exits would close it
    # on its first evaluation, at a loss, for no visible reason.
    strat_map = st["strategy_by_symbol"]

    closed_today = set()
    entry_dates = broker.entry_dates() if positions else {}
    for p in positions:
        sym = p["symbol"]
        df = bars.get(sym)
        if df is None:
            continue

        bars_held = None
        if sym in entry_dates:
            try:
                since = pd.Timestamp(entry_dates[sym])
                bars_held = int((df.index > since).sum())
            except Exception:
                bars_held = None

        opened_by = strat_map.get(sym, "pullback")
        if sym not in strat_map:
            # Positions opened before the agent recorded this have no entry in
            # the map, and pullback was the only strategy that existed then, so
            # the fallback is correct rather than a fault. Worth printing, not
            # worth marking the run unhealthy over.
            print(f"  {sym}: no record of which strategy opened it, using the "
                  f"pullback exits")
        reason = exit_decision(df, cfg.strategy, bars_held, opened_by)
        if not reason:
            continue

        try:
            fill = broker.close_position(sym, allow_live=cfg.allow_live_trading,
                                         acknowledged=args.i_understand_the_risk)
        except BrokerError as exc:
            # One symbol failing to close is not a reason to abandon the other
            # positions, skip the dashboard, and leave no record of the run.
            print(f"  {sym}: EXIT FAILED, {exc}")
            st["errors"].append(f"could not close {sym}: {exc}")
            continue
        print(f"  {sym}: CLOSING, {reason}")
        st["exits"].append({"symbol": sym, "reason": reason,
                            "status": fill.status,
                            "unrealized_pl": p.get("unrealized_pl")})
        if fill.submitted:
            # Deliberately NOT removed from `held`. The exits and the entries
            # read the same bar, so the same close that closed a mean reversion
            # trade can be a breakout signal, and the run would send a market
            # sell and a market buy for one symbol into the same open.
            closed_today.add(sym)
            strat_map.pop(sym, None)
            gross -= p.get("market_value", 0)
            open_count = max(0, open_count - 1)

    returns = pd.DataFrame({s: d["close"].pct_change() for s, d in bars.items()})
    committed = set(held)   # grows as this run takes positions

    # Collect every candidate first, then rank them, because there are almost
    # always more setups than free slots and whatever decides that ordering
    # matters more than the entry rules do. The backtest uses this same
    # function, so the two agree on which trades they would take.
    # Symbols whose data is not good enough to trade on. Flagging these and
    # then scanning them anyway would mean the warning and the behaviour
    # disagree, which is the same as having no warning.
    unfit = watch.unfit_for_trading(bars, cfg.watchlist)
    if unfit:
        print(f"Skipping {len(unfit)} symbol(s) with unusable data: "
              f"{', '.join(sorted(unfit)[:10])}"
              + (" ..." if len(unfit) > 10 else "") + "\n")

    candidates = []
    for sym, df in sorted(bars.items()):
        if sym in held or sym in unfit or sym in closed_today:
            continue
        sig = latest_signal(sym, df, cfg.strategy)
        if sig is None:
            continue
        prepared = prepare(df, cfg.strategy)
        candidates.append((sym, sig, ranking.score(
            cfg.strategy.rank_by, prepared, len(prepared) - 1, sig.stop)))

    if criticals:
        candidates = []
    ranked = ranking.order([(c[0], c[2]) for c in candidates], cfg.strategy.rank_by)
    by_symbol = {c[0]: c[1] for c in candidates}
    if cfg.strategy.rank_by != "none" and len(ranked) > 1:
        print(f"Ranking {len(ranked)} candidates by {cfg.strategy.rank_by}: "
              f"{', '.join(ranked)}\n")

    for sym in ranked:
        sig = by_symbol[sym]
        df = bars[sym]

        order = size_position(equity, sig.reference_close, sig.stop, cfg.risk,
                              cfg.strategy, open_positions=open_count,
                              gross_exposure=gross, halted=halted)
        if not order.ok:
            print(f"  {sym}: skipped, {order.rejected_reason}")
            st["skipped"].append({"symbol": sym, "reason": order.rejected_reason})
            continue

        dup = correlation_block(sym, committed, returns,
                                cfg.risk.correlation_window,
                                cfg.risk.max_correlation)
        if dup:
            print(f"  {sym}: skipped, {dup}")
            st["skipped"].append({"symbol": sym, "reason": dup})
            continue

        verdict = screen(sym, order.entry, order.stop, order.target,
                         researcher, cfg.research)
        if verdict.veto:
            print(f"  {sym}: BLOCKED by research, {verdict.reason}")
            st["vetoes"].append({"symbol": sym, "reason": verdict.reason,
                                 "flags": verdict.flags, "source": verdict.source})
            continue

        st["signals"].append({"symbol": sym, "entry": order.entry, "stop": order.stop,
                              "target": order.target, "shares": order.shares,
                              "strategy": sig.strategy, "why": sig.notes})

        try:
            fill = broker.submit_bracket(
                sym, order.shares, order.stop, order.target,
                allow_live=cfg.allow_live_trading,
                acknowledged=args.i_understand_the_risk,
            )
        except BrokerError as exc:
            # Insufficient buying power, a halted symbol, a wash trade block.
            # The broker refuses one order; the run continues to the next
            # candidate and still writes its state and its dashboard.
            print(f"  {sym}: REJECTED by the broker, {exc}")
            st["errors"].append(f"order rejected for {sym}: {exc}")
            st["skipped"].append({"symbol": sym, "reason": f"broker rejected: {exc}"})
            continue
        print(f"  {sym}: [{sig.strategy}] {fill.status} {fill.detail or ''} "
              f"({order.shares} sh, risking ${order.dollars_at_risk:,.2f})")
        # The caps move for every candidate this run commits to, filled or not.
        # Advancing them only on a real submission meant a dry run sized all of
        # them against the opening position count and exposure, so
        # `python agent.py run` printed a plan of twenty entries that a
        # --submit run would never take.
        committed.add(sym)
        open_count += 1
        gross += order.notional
        if fill.submitted:
            st["orders"].append({"symbol": sym, "shares": order.shares,
                                 "stop": order.stop, "target": order.target,
                                 "dollars_at_risk": order.dollars_at_risk,
                                 "order_id": fill.order_id,
                                 "strategy": sig.strategy})
            strat_map[sym] = sig.strategy

    if not (st["orders"] or st["signals"] or st["vetoes"]):
        print("  No setups met the rules today.")

    try:
        st["recent_trades"] = broker.realized_trades(limit=20)
    except BrokerError as exc:
        st["errors"].append(f"could not read trade history: {exc}")

    # Against the last DIFFERENT day. On a second run of the same day
    # history[-1] is this morning's own row, so the briefing was told the
    # account was flat at +0.00% whatever it had actually done.
    history = state.load_equity_history()
    day_change = 0.0
    _prior = [h for h in history if str(h.get("date", ""))[:10] != _today]
    if _prior and _prior[-1]["equity"] > 0:
        day_change = (equity / _prior[-1]["equity"] - 1) * 100

    if cfg.research.write_briefing:
        st["briefing"] = researcher.daily_briefing(
            positions, st["signals"], st["vetoes"], equity, day_change)

    # Prune to what is still committed so the map cannot grow forever with
    # symbols that were closed months ago.
    st["strategy_by_symbol"] = {k: v for k, v in strat_map.items()
                               if k in held or k in committed}

    st["research"]["llm_calls"] = researcher.calls
    st["research"]["llm_errors"] = researcher.errors
    st["healthy"] = not st["errors"]

    state.append_equity(equity, cash, open_count)
    state.save_state(st)
    state.log_run({"mode": st["mode"], "equity": equity,
                   "orders": len(st["orders"]), "vetoes": len(st["vetoes"]),
                   "submitted": bool(args.submit),
                   "healthy": bool(st["healthy"]),
                   "errors": len(st["errors"])})
    page = write_dashboard()

    if st["briefing"]:
        print(f"\n--- Briefing ---\n{st['briefing']}\n")
    print(f"{len(st['orders'])} order(s) submitted." if args.submit
          else "Dry run. Nothing was sent. Add --submit to place paper orders.")
    print(f"Dashboard: {page}")


def cmd_compare(args):
    """Run several documented strategies on the same data and print the result."""
    from tbot import compare as cmp

    cfg = build_config(args)
    symbols = sorted(set(cfg.watchlist) | set(cmp.EXTRA_ASSETS))
    print(f"Loading {len(symbols)} symbols from {args.start} "
          f"(this downloads a few years of data, give it a minute)...")
    bars = datamod.load_universe(symbols, start=args.start, end=args.end,
                                 refresh=args.refresh)
    print(f"Loaded {len(bars)}.\n")

    # Run your existing rules through the trade engine so they appear in the
    # same table, measured the same way.
    pullback_equity = None
    try:
        stock_bars = {s: d for s, d in bars.items() if s in cfg.watchlist}
        if stock_bars:
            res = run_backtest(stock_bars, cfg)
            if len(res.equity):
                pullback_equity = res.equity["equity"]
    except Exception as exc:
        print(f"(could not include your pullback rules: {exc})")

    entries = cmp.run_all(bars, cfg.watchlist,
                          slippage_bps=cfg.costs.slippage_bps,
                          start_equity=cfg.risk.starting_equity,
                          pullback_equity=pullback_equity)
    if not entries:
        print("Nothing ran. Check that SPY, EFA, AGG and SHY downloaded.")
        return

    idx = entries[0].sim.index
    split = args.split or str(idx[int(len(idx) * 0.55)].date())

    short = sorted(s for s, d in bars.items()
                   if d.index[0] > pd.Timestamp(args.start) + pd.Timedelta(days=45))
    if short:
        tail = "" if args.refresh else " Re-run with --refresh if that looks wrong."
        print(f"NOTE: {len(short)} symbol(s) start later than {args.start}: "
              f"{', '.join(short)}. Usually that just means the company or fund "
              f"did not exist yet.{tail}\n")

    full = cmp.table(entries)
    ins = cmp.table(entries, end=split)
    oos = cmp.table(entries, start=split)

    cmp.print_table(full, f"FULL PERIOD  {idx[0].date()} to {idx[-1].date()}")
    cmp.print_table(oos, f"OUT OF SAMPLE  {split} onward  <- weight this one most")
    cmp.print_table(ins, f"EARLIER PERIOD  up to {split}")

    path = rep.REPORTS / "comparison.html"
    path.write_text(cmp.compare_html(entries, full, ins, oos, split))
    print(f"\nReport: {path}")
    print("\nRead the out-of-sample table first. A strategy that wins the full")
    print("period but not that one was fitted to history you already knew.")


def cmd_journal(args):
    """What the closed trades say about which setups worked."""
    from tbot import journal

    src = rep.REPORTS / "trades.csv"
    live = state.STATE_DIR / "journal.csv"

    frames = []
    if args.source in ("both", "live") and live.exists():
        frames.append(pd.read_csv(live))
    if args.source in ("both", "backtest") and src.exists():
        frames.append(pd.read_csv(src))

    if not frames:
        print("No closed trades to analyze yet.")
        print("Run `python agent.py backtest` first, or wait for live trades to close.")
        return

    df = pd.concat(frames, ignore_index=True)
    print(journal.format_report(journal.analyze(df)))


def cmd_evolve(args):
    """Test whether re-tuning the settings actually beats leaving them alone."""
    from tbot import evolve

    cfg = build_config(args)
    print(f"Loading {len(cfg.watchlist)} symbols from {args.start}...")
    bars = datamod.load_universe(cfg.watchlist, start=args.start,
                                 refresh=args.refresh)
    print(f"Loaded {len(bars)}. Running walk-forward test, this takes "
          f"a few minutes.\n")

    params = ([p.strip() for p in args.params.split(",")] if args.params
              else list(evolve.SEARCH_SPACE.keys()))
    bad = [p for p in params if p not in evolve.SEARCH_SPACE]
    if bad:
        print(f"Not tunable: {bad}. Allowed: {list(evolve.SEARCH_SPACE)}")
        return

    seen = []
    def progress(msg):
        if msg not in seen:
            seen.append(msg)
            print(f"  {msg}")

    results = evolve.walk_forward(bars, cfg, params=params,
                                  n_windows=args.windows,
                                  train_years=args.train_years,
                                  test_years=args.test_years,
                                  progress=progress)
    v = evolve.verdict(results)
    print(evolve.format_report(results, v))


def cmd_dashboard(args):
    page = write_dashboard()
    print(f"Wrote {page}")
    if args.open:
        import subprocess
        subprocess.call(["open", str(page)])


def cmd_status(args):
    load_dotenv()
    broker = AlpacaBroker(dry_run=True)
    try:
        acct = broker.account()
    except BrokerError as exc:
        print(f"Cannot reach the broker: {exc}")
        return
    print(f"Mode          {acct['mode']}")
    print(f"Status        {acct['status']}")
    print(f"Equity        ${acct['equity']:,.2f}")
    print(f"Cash          ${acct['cash']:,.2f}")
    print(f"Buying power  ${acct['buying_power']:,.2f}")
    print(f"PDT flag      {acct['pattern_day_trader']}")

    pos = broker.positions()
    if not pos:
        print("\nNo open positions.")
        return
    print(f"\n{len(pos)} open position(s):")
    for p in pos:
        print(f"  {p['symbol']:6s} {p['shares']:>5} sh @ ${p['avg_entry']:,.2f}  "
              f"value ${p['market_value']:,.2f}  P&L ${p['unrealized_pl']:+,.2f}")

    # The order book, in the terms that decide whether a position is protected.
    # "UNPROTECTED: no stop order behind X" says what the agent concluded but
    # not what it saw, and those are different questions when the conclusion is
    # wrong. Only a stop counts, so a symbol can be reserved down to zero
    # available shares -- which is what makes a repair stop bounce with a 403
    # insufficient-qty -- while still having nothing that would actually exit.
    try:
        orders = broker.open_orders()
    except BrokerError as exc:
        print(f"\nCannot read the order book: {exc}")
        return

    flat = watch._flatten_orders(orders)
    stop_qty, sells = watch._stop_coverage(orders)
    exposed = watch.unprotected(pos, orders)

    print(f"\n{len(flat)} working order(s) at the broker:")
    if not flat:
        print("  none")
    for o in sorted(flat, key=lambda x: (str(x.get('symbol')), str(x.get('id')))):
        counts = ("STOP" if o.get("stop_price") not in (None, "")
                  and str(o.get("side", "")).startswith("sell") else "not a stop")
        print(f"  {str(o.get('symbol')):6s} {str(o.get('side')):5s} "
              f"{str(o.get('type') or o.get('order_type')):11s} "
              f"qty {str(o.get('qty')):>6s}  "
              f"stop {str(o.get('stop_price') or '-'):>9s}  "
              f"limit {str(o.get('limit_price') or '-'):>9s}  "
              f"{str(o.get('status')):16s} {counts}")

    print("\nProtection, per position:")
    for p in pos:
        sym = p["symbol"]
        gap = exposed.get(sym, 0)
        verdict = (f"EXPOSED {gap} of {p['shares']} sh" if gap
                   else f"covered ({stop_qty.get(sym, 0)} sh under a stop)")
        print(f"  {sym:6s} {verdict}   "
              f"[{sells.get(sym, 0)} sell order(s) seen on this symbol]")

    if exposed:
        print("\nA symbol shown here with 0 sell orders seen, that the broker "
              "still refuses a stop on for insufficient quantity, means the "
              "shares are reserved by an order this list did not return -- the "
              "position is protected and the alarm is wrong. A symbol with 1 "
              "sell order that is 'not a stop' means the take-profit leg "
              "outlived its stop, and the position really is exposed.")


def cmd_cache(args):
    df = datamod.cache_status()
    if df.empty:
        print("Cache is empty. Run a backtest or scan to populate it.")
        return
    print(df.to_string(index=False))


def cmd_selftest(args):
    import subprocess
    here = Path(__file__).resolve().parent
    sys.exit(subprocess.call([sys.executable, "-m", "tests.test_logic"], cwd=here))


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Trend pullback trading agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, start="2018-01-01"):
        sp.add_argument("--symbols", help="comma separated, overrides the watchlist")
        sp.add_argument("--start", default=start)
        sp.add_argument("--equity", type=float, help="account size for sizing")
        sp.add_argument("--risk", type=float, help="percent risked per trade, e.g. 1.0")
        sp.add_argument("--refresh", action="store_true", help="re-download market data")

    b = sub.add_parser("backtest", help="test the rules on history")
    common(b)
    b.add_argument("--end", default=None)
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("scan", help="today's signals, no orders")
    common(s, start="2023-01-01")
    s.add_argument("--use-broker", action="store_true",
                   help="size off your real account equity")
    s.set_defaults(func=cmd_scan)

    for name, helptext in (("run", "the daily job: scan, screen, trade, record"),
                           ("paper", "alias for run")):
        pa = sub.add_parser(name, help=helptext)
        common(pa, start="2023-01-01")
        pa.add_argument("--submit", action="store_true", help="actually send the orders")
        pa.add_argument("--no-research", action="store_true",
                        help="skip the earnings filter and news review")
        pa.add_argument("--i-understand-the-risk", action="store_true",
                        help="third safety lock, required only for live accounts")
        pa.add_argument("--ignore-session", action="store_true",
                        help="run even while the market is open, on an "
                             "unfinished bar (for testing only)")
        pa.set_defaults(func=cmd_run)

    mo = sub.add_parser("monitor",
                        help="mid-session safety check: no entries, no exits")
    common(mo, start="2023-01-01")
    mo.add_argument("--submit", action="store_true",
                    help="actually send the protective stops it finds missing")
    mo.add_argument("--i-understand-the-risk", action="store_true",
                    help="third safety lock, required only for live accounts")
    mo.set_defaults(func=cmd_monitor)

    cp = sub.add_parser("compare", help="race several strategies against each other")
    common(cp, start="2005-01-01")
    cp.add_argument("--end", default=None)
    cp.add_argument("--split", default=None,
                    help="out-of-sample start date, e.g. 2016-01-01")
    cp.set_defaults(func=cmd_compare)

    j = sub.add_parser("journal", help="what your closed trades actually show")
    j.add_argument("--source", choices=["both", "live", "backtest"], default="both")
    j.set_defaults(func=cmd_journal)

    ev = sub.add_parser("evolve", help="test whether re-tuning beats leaving it alone")
    common(ev, start="2010-01-01")
    ev.add_argument("--params", help="comma separated, default all tunable ones")
    ev.add_argument("--windows", type=int, default=4)
    ev.add_argument("--train-years", type=float, default=3.0)
    ev.add_argument("--test-years", type=float, default=1.0)
    ev.set_defaults(func=cmd_evolve)

    d = sub.add_parser("dashboard", help="rebuild the dashboard page")
    d.add_argument("--open", action="store_true", help="open it in your browser")
    d.set_defaults(func=cmd_dashboard)

    st = sub.add_parser("status", help="account and open positions")
    st.set_defaults(func=cmd_status)

    c = sub.add_parser("cache", help="show cached market data")
    c.set_defaults(func=cmd_cache)

    t = sub.add_parser("selftest", help="verify the engine's math")
    t.set_defaults(func=cmd_selftest)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
