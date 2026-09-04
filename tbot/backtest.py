"""
Event-driven daily backtester.

Design rules, because these are where backtests usually cheat:

  * A signal is generated from a COMPLETED bar and filled at the NEXT bar's
    open. Never at the signal bar's close.
  * If the next open gaps through your stop, you fill at the open, not at the
    stop. Gaps are the single most common way a "1% risk" trade turns into a
    4% loss, and a backtest that ignores them is lying to you.
  * If a bar touches both the stop and the target, the stop is assumed to hit
    first. Daily bars do not tell you the order of events inside the day, so
    we take the pessimistic reading every time.
  * Slippage and commissions are charged on every fill.
  * Positions are sized off the equity at the moment of entry, not off
    hindsight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import AgentConfig
from .risk import (DrawdownMonitor, apply_slippage, commission,
                   correlation_block, size_position)
from .strategy import prepare, _row_signal


@dataclass
class Trade:
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry: float
    stop: float
    target: float
    shares: int
    exit_date: Optional[pd.Timestamp] = None
    exit: Optional[float] = None
    exit_reason: str = ""
    fees: float = 0.0

    @property
    def initial_risk_per_share(self) -> float:
        return self.entry - self.stop

    @property
    def pnl(self) -> float:
        if self.exit is None:
            return 0.0
        return (self.exit - self.entry) * self.shares - self.fees

    @property
    def r_multiple(self) -> float:
        """Profit measured in units of what you risked. The only honest way
        to compare a win on TSLA to a win on SPY."""
        risk = self.initial_risk_per_share * self.shares
        if risk <= 0 or self.exit is None:
            return 0.0
        return ((self.exit - self.entry) * self.shares - self.fees) / risk

    @property
    def bars_held(self) -> int:
        if self.exit_date is None:
            return 0
        return int(np.busday_count(self.entry_date.date(), self.exit_date.date()))


@dataclass
class Position:
    trade: Trade
    bars_held: int = 0
    pending_exit: bool = False
    pending_exit_reason: str = ""


@dataclass
class PendingEntry:
    symbol: str
    signal_date: pd.Timestamp
    stop: float


class Backtest:
    def __init__(self, bars: Dict[str, pd.DataFrame], cfg: AgentConfig):
        self.cfg = cfg
        self.raw = bars
        self.data = {s: prepare(df, cfg.strategy) for s, df in bars.items()}

        self.cash = cfg.risk.starting_equity
        self.positions: Dict[str, Position] = {}
        self.pending: Dict[str, PendingEntry] = {}
        self.closed: List[Trade] = []
        self.equity_curve: List[dict] = []
        self.dd = DrawdownMonitor(cfg.risk.starting_equity,
                                  cfg.risk.max_drawdown_halt,
                                  cfg.risk.resume_below,
                                  cfg.risk.halt_cooldown_days)
        self.halt_days = 0

        all_dates = sorted(set().union(*[set(df.index) for df in self.data.values()]))
        self.calendar = pd.DatetimeIndex(all_dates)

        # Position-indexed lookup so we can grab bar t and bar t-1 in O(1).
        self._pos = {s: {d: i for i, d in enumerate(df.index)} for s, df in self.data.items()}

        # Daily returns matrix, used to refuse trades that duplicate a
        # position you already hold.
        self.returns = pd.DataFrame(
            {s: df["close"].pct_change() for s, df in self.data.items()}
        ).sort_index()

    # -- helpers ------------------------------------------------------------

    def _bar(self, sym: str, date: pd.Timestamp):
        i = self._pos[sym].get(date)
        if i is None:
            return None, None, None
        df = self.data[sym]
        prev = df.iloc[i - 1] if i > 0 else None
        return df.iloc[i], prev, i

    def _equity(self, date: pd.Timestamp) -> float:
        value = self.cash
        for sym, pos in self.positions.items():
            row, _, _ = self._bar(sym, date)
            price = float(row["close"]) if row is not None else pos.trade.entry
            value += pos.trade.shares * price
        return value

    def _gross_exposure(self) -> float:
        return sum(p.trade.shares * p.trade.entry for p in self.positions.values())

    # -- fills --------------------------------------------------------------

    def _close_position(self, sym: str, date, price: float, reason: str):
        pos = self.positions.pop(sym)
        t = pos.trade
        fee = commission(t.shares, self.cfg.costs)
        t.exit = round(price, 4)
        t.exit_date = date
        t.exit_reason = reason
        t.fees += fee
        self.cash += t.shares * price - fee
        self.closed.append(t)

    def _process_exits(self, date: pd.Timestamp):
        costs = self.cfg.costs
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            row, _, _ = self._bar(sym, date)
            if row is None:
                continue

            t = pos.trade
            o, h, l = float(row["open"]), float(row["high"]), float(row["low"])

            # A queued exit (trend break or time stop) fills at today's open.
            if pos.pending_exit:
                fill = apply_slippage(o, "sell", costs.slippage_bps)
                self._close_position(sym, date, fill, pos.pending_exit_reason)
                continue

            # Gap down straight through the stop: you fill at the open, worse
            # than your stop. This is the risk that position sizing cannot
            # protect you from, and it is why we cap position size separately.
            if o <= t.stop:
                fill = apply_slippage(o, "sell", costs.stop_slippage_bps)
                self._close_position(sym, date, fill, "gap through stop")
                continue

            # Gap up through the target: you fill at the open, better than
            # your target.
            if o >= t.target:
                fill = apply_slippage(o, "sell", costs.slippage_bps)
                self._close_position(sym, date, fill, "gap through target")
                continue

            # Intraday. If both levels were touched we assume the stop hit
            # first, because a daily bar cannot tell us otherwise.
            if l <= t.stop:
                fill = apply_slippage(t.stop, "sell", costs.stop_slippage_bps)
                self._close_position(sym, date, fill, "stop")
                continue

            if h >= t.target:
                fill = apply_slippage(t.target, "sell", costs.slippage_bps)
                self._close_position(sym, date, fill, "target")
                continue

            # Still open. Age it and check the soft exits, which fire tomorrow.
            pos.bars_held += 1
            c = float(row["close"])
            sma_fast = row["sma_fast"]

            if self.cfg.strategy.exit_on_trend_break and not pd.isna(sma_fast) and c < float(sma_fast):
                pos.pending_exit = True
                pos.pending_exit_reason = "trend break"
            elif pos.bars_held >= self.cfg.strategy.max_hold_days:
                pos.pending_exit = True
                pos.pending_exit_reason = "time stop"

    # -- entries ------------------------------------------------------------

    def _process_entries(self, date: pd.Timestamp):
        costs = self.cfg.costs
        equity = self._equity(date)
        halted = self.dd.tripped

        for sym in list(self.pending.keys()):
            pend = self.pending.pop(sym)
            if sym in self.positions:
                continue

            row, _, _ = self._bar(sym, date)
            if row is None:
                continue

            fill = apply_slippage(float(row["open"]), "buy", costs.slippage_bps)

            # The stop was set from the signal bar. If the stock gapped open
            # below it, the setup is void before it started.
            if fill <= pend.stop:
                continue

            # Refuse a near-duplicate of something already held. Only data up
            # to the previous close is used, so this cannot see the future.
            if self.positions:
                hist = self.returns[self.returns.index < date]
                if correlation_block(sym, list(self.positions.keys()), hist,
                                     self.cfg.risk.correlation_window,
                                     self.cfg.risk.max_correlation):
                    continue

            order = size_position(
                equity=equity,
                entry=fill,
                stop=pend.stop,
                cfg_risk=self.cfg.risk,
                cfg_strategy=self.cfg.strategy,
                open_positions=len(self.positions),
                gross_exposure=self._gross_exposure(),
                halted=halted,
            )
            if not order.ok:
                continue

            fee = commission(order.shares, costs)
            cost = order.shares * fill + fee
            if cost > self.cash:
                continue

            self.cash -= cost
            self.positions[sym] = Position(
                trade=Trade(
                    symbol=sym,
                    signal_date=pend.signal_date,
                    entry_date=date,
                    entry=round(fill, 4),
                    stop=order.stop,
                    target=order.target,
                    shares=order.shares,
                    fees=fee,
                )
            )

    # -- signals ------------------------------------------------------------

    def _scan(self, date: pd.Timestamp):
        for sym, df in self.data.items():
            if sym in self.positions or sym in self.pending:
                continue
            row, prev, i = self._bar(sym, date)
            if row is None or prev is None:
                continue
            sig = _row_signal(sym, date, row, prev, self.cfg.strategy)
            if sig is not None:
                self.pending[sym] = PendingEntry(sym, date, sig.stop)

    # -- main loop ----------------------------------------------------------

    def run(self) -> "BacktestResult":
        for date in self.calendar:
            self._process_exits(date)
            self._process_entries(date)

            equity = self._equity(date)
            if self.dd.update(equity):
                self.halt_days += 1

            self._scan(date)

            self.equity_curve.append({
                "date": date,
                "equity": equity,
                "cash": self.cash,
                "open_positions": len(self.positions),
                "drawdown": self.dd.drawdown(equity),
            })

        # Close whatever is still open at the last close, so the numbers are
        # not flattered by unrealized winners.
        if self.calendar.size:
            last = self.calendar[-1]
            for sym in list(self.positions.keys()):
                row, _, _ = self._bar(sym, last)
                price = float(row["close"]) if row is not None else self.positions[sym].trade.entry
                self._close_position(sym, last, price, "end of test")

        return BacktestResult(self.closed, pd.DataFrame(self.equity_curve).set_index("date"),
                              self.cfg, self.halt_days)


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity: pd.DataFrame
    cfg: AgentConfig
    halt_days: int = 0

    def trades_df(self) -> pd.DataFrame:
        rows = []
        for t in self.trades:
            rows.append({
                "symbol": t.symbol,
                "signal_date": t.signal_date.date(),
                "entry_date": t.entry_date.date(),
                "exit_date": t.exit_date.date() if t.exit_date else None,
                "shares": t.shares,
                "entry": round(t.entry, 2),
                "stop": round(t.stop, 2),
                "target": round(t.target, 2),
                "exit": round(t.exit, 2) if t.exit else None,
                "reason": t.exit_reason,
                "pnl": round(t.pnl, 2),
                "R": round(t.r_multiple, 2),
                "days": t.bars_held,
            })
        return pd.DataFrame(rows)

    def stats(self) -> dict:
        df = self.trades_df()
        eq = self.equity["equity"]
        start = self.cfg.risk.starting_equity
        end = float(eq.iloc[-1]) if len(eq) else start

        if df.empty:
            return {"trades": 0, "note": "no trades generated"}

        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]

        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9) if len(eq) > 1 else 1e-9
        cagr = (end / start) ** (1 / years) - 1 if start > 0 and end > 0 else float("nan")

        daily_ret = eq.pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else float("nan")

        gross_win = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())

        # Time in the market. A strategy that is flat 80% of the time is not
        # comparable to buy and hold, even if the returns look similar.
        exposure = (self.equity["open_positions"] > 0).mean()

        return {
            "start_equity": round(start, 2),
            "end_equity": round(end, 2),
            "total_return_pct": round((end / start - 1) * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "max_drawdown_pct": round(self.equity["drawdown"].max() * 100, 2),
            "sharpe": round(float(sharpe), 2),
            "trades": len(df),
            "win_rate_pct": round(len(wins) / len(df) * 100, 1),
            "avg_win": round(wins["pnl"].mean(), 2) if len(wins) else 0.0,
            "avg_loss": round(losses["pnl"].mean(), 2) if len(losses) else 0.0,
            "expectancy_R": round(df["R"].mean(), 3),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
            "avg_days_held": round(df["days"].mean(), 1),
            "exposure_pct": round(exposure * 100, 1),
            "halt_days": self.halt_days,
            "exit_breakdown": df["reason"].value_counts().to_dict(),
        }


def run_backtest(bars: Dict[str, pd.DataFrame], cfg: AgentConfig) -> BacktestResult:
    return Backtest(bars, cfg).run()
