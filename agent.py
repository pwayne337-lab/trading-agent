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
import os
import sys
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
from tbot.risk import correlation_block, size_position
from tbot.strategy import latest_signal, trend_state


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


def cmd_run(args):
    """The daily job: rules find setups, research screens them, orders go in,
    everything is recorded, the dashboard is rebuilt."""
    load_dotenv()
    cfg = build_config(args)
    if args.no_research:
        cfg.research.use_llm = False
        cfg.research.check_earnings = False

    broker = AlpacaBroker(dry_run=not args.submit)
    st = state.blank_state()
    st["mode"] = "live" if broker.is_live else "paper"

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
        state.save_state(st)
        # Rebuild the page even on failure, so the dashboard reports the
        # outage instead of silently showing yesterday's numbers as current.
        write_dashboard()
        return

    equity, cash = acct["equity"], acct["cash"]

    # Count orders that are accepted but not yet filled as already owned.
    # Without this, a second run before the market opens submits the same
    # trade again and silently doubles the risk on it.
    try:
        orders_open = broker.open_orders()
    except BrokerError as exc:
        orders_open = []
        st["errors"].append(f"could not read working orders: {exc}")

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
        state.save_state(st)
        print("Trading is blocked on this account. Exiting.")
        return

    researcher = Researcher(enabled=cfg.research.use_llm)
    st["research"]["enabled"] = researcher.enabled
    if cfg.research.use_llm and not researcher.enabled:
        print("No ANTHROPIC_API_KEY found, so news review is off. "
              "The earnings filter still runs.\n")

    bars = datamod.load_universe(cfg.watchlist, start=args.start, refresh=True)

    returns = pd.DataFrame({s: d["close"].pct_change() for s, d in bars.items()})
    committed = set(held)   # grows as this run takes positions

    for sym, df in sorted(bars.items()):
        if sym in held:
            continue
        sig = latest_signal(sym, df, cfg.strategy)
        if sig is None:
            continue

        order = size_position(equity, sig.reference_close, sig.stop, cfg.risk,
                              cfg.strategy, open_positions=open_count,
                              gross_exposure=gross)
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
                              "target": order.target, "shares": order.shares})

        fill = broker.submit_bracket(
            sym, order.shares, order.stop, order.target,
            allow_live=cfg.allow_live_trading,
            acknowledged=args.i_understand_the_risk,
        )
        print(f"  {sym}: {fill.status} {fill.detail or ''} "
              f"({order.shares} sh, risking ${order.dollars_at_risk:,.2f})")
        committed.add(sym)
        if fill.submitted:
            st["orders"].append({"symbol": sym, "shares": order.shares,
                                 "stop": order.stop, "target": order.target,
                                 "dollars_at_risk": order.dollars_at_risk,
                                 "order_id": fill.order_id})
            open_count += 1
            gross += order.notional

    if not (st["orders"] or st["signals"] or st["vetoes"]):
        print("  No setups met the rules today.")

    try:
        st["recent_trades"] = broker.realized_trades(limit=20)
    except BrokerError as exc:
        st["errors"].append(f"could not read trade history: {exc}")

    history = state.load_equity_history()
    day_change = 0.0
    if history and history[-1]["equity"] > 0:
        day_change = (equity / history[-1]["equity"] - 1) * 100

    if cfg.research.write_briefing:
        st["briefing"] = researcher.daily_briefing(
            positions, st["signals"], st["vetoes"], equity, day_change)

    st["research"]["llm_calls"] = researcher.calls
    st["research"]["llm_errors"] = researcher.errors
    st["healthy"] = not st["errors"]

    state.append_equity(equity, cash, open_count)
    state.save_state(st)
    state.log_run({"mode": st["mode"], "equity": equity,
                   "orders": len(st["orders"]), "vetoes": len(st["vetoes"]),
                   "submitted": bool(args.submit)})
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
        pa.set_defaults(func=cmd_run)

    cp = sub.add_parser("compare", help="race several strategies against each other")
    common(cp, start="2005-01-01")
    cp.add_argument("--end", default=None)
    cp.add_argument("--split", default=None,
                    help="out-of-sample start date, e.g. 2016-01-01")
    cp.set_defaults(func=cmd_compare)

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
