"""
Position sizing and portfolio risk.

This is the part of an automated trading system that actually decides whether
you survive. The entry logic decides which trades you take. This decides how
much a wrong one costs. Beginners spend all their time on the first and none
on the second.

Fixed fractional sizing, the method used here:

    shares = (equity * risk_per_trade) / (entry - stop)

If you have $10,000, risk 1% ($100), and your stop is $2 below your entry,
you buy 50 shares. If the stop is $0.50 below entry, you buy 200 shares. The
dollar loss on a stop-out is the same either way. That is the entire point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class SizedOrder:
    shares: int
    entry: float
    stop: float
    target: float
    risk_per_share: float
    dollars_at_risk: float
    notional: float
    pct_of_equity: float
    rejected_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.rejected_reason is None and self.shares > 0


def size_position(equity: float, entry: float, stop: float, cfg_risk, cfg_strategy,
                  open_positions: int = 0, gross_exposure: float = 0.0,
                  halted: bool = False) -> SizedOrder:
    """Turn a price and a stop into a share count, or refuse the trade."""

    risk_per_share = entry - stop
    empty = SizedOrder(0, entry, stop, 0.0, risk_per_share, 0.0, 0.0, 0.0)

    if halted:
        empty.rejected_reason = "drawdown circuit breaker active"
        return empty

    if risk_per_share <= 0:
        empty.rejected_reason = "stop is at or above entry"
        return empty

    if open_positions >= cfg_risk.max_open_positions:
        empty.rejected_reason = f"already holding {open_positions} positions (max {cfg_risk.max_open_positions})"
        return empty

    dollars_to_risk = equity * cfg_risk.risk_per_trade
    raw_shares = dollars_to_risk / risk_per_share

    # Cap by position size. A very tight stop would otherwise size you into a
    # position worth more than the account.
    max_notional = equity * cfg_risk.max_position_pct
    shares_by_notional = max_notional / entry

    # Cap by remaining buying power. No margin.
    room = max(0.0, equity * cfg_risk.max_gross_exposure - gross_exposure)
    shares_by_room = room / entry

    shares = int(math.floor(min(raw_shares, shares_by_notional, shares_by_room)))

    if shares < 1:
        empty.rejected_reason = "position would be less than one share"
        return empty

    target = entry + cfg_strategy.reward_risk * risk_per_share
    notional = shares * entry

    return SizedOrder(
        shares=shares,
        entry=round(entry, 4),
        stop=round(stop, 4),
        target=round(target, 4),
        risk_per_share=round(risk_per_share, 4),
        dollars_at_risk=round(shares * risk_per_share, 2),
        notional=round(notional, 2),
        pct_of_equity=round(notional / equity * 100, 2) if equity > 0 else 0.0,
    )


def correlation_block(candidate: str, held, returns, window: int,
                      max_corr: float):
    """Return a reason string if the candidate duplicates something held.

    `returns` is a frame of daily returns with one column per symbol. We look
    at the most recent `window` rows and compare the candidate against every
    open position.

    Why this exists: five positions that all move together is one position
    with five times the size. Position sizing assumes a stop-out costs 1% of
    the account. If everything you hold falls on the same day, five 1% risks
    become a single 5% loss, which is the sizing you thought you were avoiding.
    """
    if returns is None or candidate not in getattr(returns, "columns", []):
        return None

    recent = returns.tail(window)
    if len(recent) < max(20, window // 3):
        return None   # not enough history to judge; do not block on noise

    a = recent[candidate]
    for other in held:
        if other == candidate or other not in recent.columns:
            continue
        pair = recent[[candidate, other]].dropna()
        if len(pair) < 20:
            continue
        c = pair[candidate].corr(pair[other])
        if c is not None and not (c != c) and c >= max_corr:
            return f"moves almost identically to {other} ({c:.2f} correlation)"
    return None


def apply_slippage(price: float, side: str, bps: float) -> float:
    """Move the fill price against you by `bps` basis points.

    Slippage is the gap between the price you saw and the price you got. It is
    small per trade and enormous over hundreds of trades. A backtest that
    assumes perfect fills is a fantasy.
    """
    factor = bps / 10_000.0
    if side == "buy":
        return price * (1.0 + factor)
    return price * (1.0 - factor)


def commission(shares: int, cfg_costs) -> float:
    return shares * cfg_costs.commission_per_share + cfg_costs.commission_per_order


class DrawdownMonitor:
    """Tracks the high water mark and trips a breaker on a deep drawdown.

    The breaker re-arms once the account has climbed back to within
    `resume_below` of its high water mark. That hysteresis matters: without
    it, one bad stretch disables the strategy permanently. In a multi-year
    backtest that silently flatlines the equity curve for the rest of the
    test and reports the result as if the strategy simply stopped making
    money, which is a lie about the strategy rather than a fact about it.
    """

    def __init__(self, starting_equity: float, limit: float,
                 resume_below: float = 0.10, cooldown_days: int = 60):
        self.peak = starting_equity
        self.limit = limit
        self.resume_below = min(resume_below, limit)
        self.cooldown_days = cooldown_days
        self.tripped = False
        self.trips = 0
        self.days_tripped = 0

    def update(self, equity: float) -> bool:
        self.peak = max(self.peak, equity)
        dd = 0.0 if self.peak <= 0 else (self.peak - equity) / self.peak

        if not self.tripped:
            if dd >= self.limit:
                self.tripped = True
                self.trips += 1
                self.days_tripped = 1     # the day it tripped counts as day one
            return self.tripped

        # Recovering to within resume_below is the good exit, and it keeps the
        # old high water mark because the loss was genuinely made back.
        if dd <= self.resume_below:
            self.tripped = False
            self.days_tripped = 0
            return False

        # The other exit. A halted strategy holds nothing, so its equity stops
        # moving, so its drawdown from the old peak never shrinks: recovery
        # alone is a deadlock, because clearing the breaker requires trading
        # and trading is what the breaker forbids. After the cooldown it
        # resumes and the high water mark resets to where the account actually
        # is. You have taken the loss; this is the new starting line, and the
        # next 20% is measured from here rather than from a peak you may never
        # see again.
        self.days_tripped += 1
        if self.days_tripped >= self.cooldown_days:
            self.tripped = False
            self.days_tripped = 0
            self.peak = equity
        return self.tripped

    def drawdown(self, equity: float) -> float:
        if self.peak <= 0:
            return 0.0
        return (self.peak - equity) / self.peak
