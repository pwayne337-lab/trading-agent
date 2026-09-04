"""
The trend pullback strategy.

Plain English version of the rules:

  1. Only look at a stock that is in an uptrend. Uptrend means the close is
     above the 200-day average AND the 50-day average is above the 200-day.
  2. Wait for a pullback. Price has to dip to or below the 20-day EMA.
  3. Buy the reclaim. When price closes back above the 20 EMA after that dip,
     that bar is the trigger.
  4. Enter at the next session's open. Not at the trigger close. You cannot
     trade a price you only learned about after the bell.
  5. Stop goes below the recent swing low, with a small ATR cushion.
  6. Target is 2x whatever you are risking.

What this strategy is NOT: it is not an edge on its own. Trend pullback is one
of the most widely traded patterns in existence. Whatever edge it has comes
from position sizing, from skipping bad setups, and from actually following it.
The backtest will show you long flat stretches and drawdowns. Those are real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .indicators import add_indicators


@dataclass
class Signal:
    """One trade idea, fully specified before any money moves."""
    symbol: str
    signal_date: pd.Timestamp     # the bar that triggered it (a completed bar)
    reference_close: float        # close of the trigger bar, for context only
    stop: float
    atr: float
    trend_ok: bool
    notes: str = ""
    strategy: str = "pullback"

    def target_from(self, entry: float, reward_risk: float) -> float:
        return entry + reward_risk * (entry - self.stop)

    def risk_per_share_from(self, entry: float) -> float:
        return entry - self.stop


def _pullback_signal(sym: str, idx: pd.Timestamp, row: pd.Series,
                     prev: pd.Series, cfg) -> Optional[Signal]:
    """Evaluate one completed bar. Returns a Signal or None.

    Uses only `row` (bar t) and `prev` (bar t-1). No future data, by
    construction, because nothing after t is passed in.
    """
    # Indicators need warmup. Until they exist there is nothing to say.
    needed = ["sma_fast", "sma_slow", "ema_pb", "atr", "swing_low", "adv"]
    if any(pd.isna(row[c]) for c in needed):
        return None
    if pd.isna(prev["ema_pb"]):
        return None

    # 1. Liquidity. Thin names cost you more in spread than the setup is worth.
    if row["adv"] < cfg.min_avg_dollar_volume:
        return None

    # 2. Trend filter.
    trend_ok = (row["close"] > row["sma_slow"]) and (row["sma_fast"] > row["sma_slow"])
    if not trend_ok:
        return None

    # 3. Pullback and reclaim, in one test. Yesterday closed at or below the
    #    20 EMA (that is the pullback), today closed above it (the reclaim).
    pulled_back = prev["close"] <= prev["ema_pb"]
    reclaimed = row["close"] > row["ema_pb"]
    if not (pulled_back and reclaimed):
        return None

    # 4. Stop under the recent swing low with an ATR cushion.
    stop = float(row["swing_low"]) - cfg.stop_atr_buffer * float(row["atr"])
    if stop <= 0:
        return None

    ref_close = float(row["close"])
    stop_distance = ref_close - stop
    if stop_distance <= 0:
        return None

    # 5. Stop distance guard rails, measured in ATR.
    atr_val = float(row["atr"])
    if atr_val <= 0:
        return None
    stop_atr = stop_distance / atr_val
    if stop_atr < cfg.min_stop_atr:
        return None   # stop is inside the noise, it will get hit at random
    if stop_atr > cfg.max_stop_atr:
        return None   # too wide to size sensibly

    return Signal(
        symbol=sym,
        signal_date=idx,
        reference_close=ref_close,
        stop=round(stop, 4),
        atr=round(atr_val, 4),
        trend_ok=True,
        notes=f"pullback to the {cfg.ema_pullback} EMA, reclaimed; "
              f"stop {stop_atr:.2f}x ATR below entry",
        strategy="pullback",
    )


# ---------------------------------------------------------------------------
# Breakout
# ---------------------------------------------------------------------------

def _breakout_signal(sym: str, idx: pd.Timestamp, row: pd.Series,
                     prev: pd.Series, cfg) -> Optional[Signal]:
    """Buy a stock making a new N-session high inside an uptrend.

    The opposite instinct from the pullback. The pullback waits for a dip and
    buys it. This waits for the stock to clear everything it has done in
    months and buys that. Both are documented, both are ordinary, and they
    fail in different weather, which is the only reason to run both.
    """
    needed = ["sma_fast", "sma_slow", "atr", "adv", "donchian_hi"]
    if any(pd.isna(row[c]) for c in needed):
        return None
    if pd.isna(prev["donchian_hi"]):
        return None

    if row["adv"] < cfg.min_avg_dollar_volume:
        return None

    if not ((row["close"] > row["sma_slow"]) and (row["sma_fast"] > row["sma_slow"])):
        return None

    # Today clears the prior range and yesterday did not. Without the second
    # half, every bar of a long advance reads as a fresh breakout and the same
    # move gets bought over and over.
    broke_out = float(row["close"]) > float(row["donchian_hi"])
    was_flat = float(prev["close"]) <= float(prev["donchian_hi"])
    if not (broke_out and was_flat):
        return None

    atr_val = float(row["atr"])
    if atr_val <= 0:
        return None

    ref_close = float(row["close"])
    stop = ref_close - cfg.breakout_stop_atr * atr_val
    if stop <= 0:
        return None

    stop_atr = (ref_close - stop) / atr_val
    if stop_atr < cfg.min_stop_atr or stop_atr > cfg.max_stop_atr:
        return None

    return Signal(
        symbol=sym,
        signal_date=idx,
        reference_close=ref_close,
        stop=round(stop, 4),
        atr=round(atr_val, 4),
        trend_ok=True,
        notes=f"new {cfg.breakout_lookback}-session high; "
              f"stop {stop_atr:.2f}x ATR below entry",
        strategy="breakout",
    )


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------

def _reversion_signal(sym: str, idx: pd.Timestamp, row: pd.Series,
                      prev: pd.Series, cfg) -> Optional[Signal]:
    """Buy a hard short-term selloff in a stock that is still healthy long term.

    This one enters while price is still falling, which is why it gets a wider
    stop and a much shorter leash than the other two. A bounce that has not
    arrived within two weeks is not arriving.
    """
    needed = ["sma_slow", "atr", "adv", "rsi_fast", "sma_exit"]
    if any(pd.isna(row[c]) for c in needed):
        return None

    if row["adv"] < cfg.min_avg_dollar_volume:
        return None

    # Long-term health only. Requiring the 50 above the 200 as well would rule
    # out most real selloffs, which is the thing this strategy exists to buy.
    if float(row["close"]) <= float(row["sma_slow"]):
        return None

    if float(row["rsi_fast"]) >= cfg.reversion_rsi_max:
        return None
    if float(row["close"]) >= float(row["sma_exit"]):
        return None

    atr_val = float(row["atr"])
    if atr_val <= 0:
        return None

    ref_close = float(row["close"])
    stop = ref_close - cfg.reversion_stop_atr * atr_val
    if stop <= 0:
        return None

    stop_atr = (ref_close - stop) / atr_val
    if stop_atr < cfg.min_stop_atr or stop_atr > cfg.max_stop_atr:
        return None

    return Signal(
        symbol=sym,
        signal_date=idx,
        reference_close=ref_close,
        stop=round(stop, 4),
        atr=round(atr_val, 4),
        trend_ok=True,
        notes=f"RSI({cfg.reversion_rsi_period}) at "
              f"{float(row['rsi_fast']):.0f}, below the {cfg.reversion_sma}-day "
              f"average; stop {stop_atr:.2f}x ATR below entry",
        strategy="reversion",
    )


def prepare(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Attach indicators to a raw OHLCV frame."""
    return add_indicators(df, cfg)


# ---------------------------------------------------------------------------
# The soft exits, one set per strategy
# ---------------------------------------------------------------------------
# These are the exits a broker cannot hold for you. The stop and the target go
# in with the entry and fire on their own at any hour. These depend on where a
# CLOSE lands, or on how long a trade has been open, so something has to look
# once a day and act. In the backtest they accounted for more than half of all
# exits, so an agent that skips them is not running the strategy that was
# tested.


def _trend_exit(row: pd.Series, cfg, bars_held: Optional[int]) -> Optional[str]:
    """For the two trend strategies: leave when the trend itself breaks."""
    if cfg.exit_on_trend_break and not pd.isna(row.get("sma_fast")):
        if float(row["close"]) < float(row["sma_fast"]):
            return (f"trend break: closed at ${float(row['close']):,.2f}, "
                    f"below the {cfg.sma_fast}-day average of "
                    f"${float(row['sma_fast']):,.2f}")
    if bars_held is not None and bars_held >= cfg.max_hold_days:
        return f"time stop: held {bars_held} sessions, limit is {cfg.max_hold_days}"
    return None


def _reversion_exit(row: pd.Series, cfg, bars_held: Optional[int]) -> Optional[str]:
    """For mean reversion: the bounce arriving IS the exit.

    Waiting for a 2:1 target here would be running a different strategy. The
    trade was taken because price was stretched below its short average, so
    the moment it is not, the reason for holding is gone.
    """
    if not pd.isna(row.get("sma_exit")):
        if float(row["close"]) > float(row["sma_exit"]):
            return (f"bounce complete: closed at ${float(row['close']):,.2f}, "
                    f"back above the {cfg.reversion_sma}-day average of "
                    f"${float(row['sma_exit']):,.2f}")
    if bars_held is not None and bars_held >= cfg.reversion_max_hold:
        return (f"time stop: held {bars_held} sessions, this strategy's limit "
                f"is {cfg.reversion_max_hold}")
    return None


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

@dataclass
class StrategySpec:
    """One complete rule set: how it gets in, how it gets out, what it is."""
    name: str
    entry: Callable[..., Optional[Signal]]
    exit: Callable[..., Optional[str]]
    summary: str


PLAYBOOK: Dict[str, StrategySpec] = {
    "pullback": StrategySpec(
        "pullback", _pullback_signal, _trend_exit,
        "Buys a dip to the 20-day EMA inside an established uptrend."),
    "breakout": StrategySpec(
        "breakout", _breakout_signal, _trend_exit,
        "Buys a new 50-session high inside an established uptrend."),
    "reversion": StrategySpec(
        "reversion", _reversion_signal, _reversion_exit,
        "Buys a two-day washout in a stock still above its 200-day average."),
}


class UnknownStrategy(KeyError):
    pass


def enabled_specs(cfg) -> List[StrategySpec]:
    """The rule sets that are switched on, in the configured order.

    An unknown name is an error rather than a silent skip. A typo in a config
    file that quietly disables a strategy is the kind of fault that shows up
    months later as "why did it stop taking those trades".
    """
    names = list(getattr(cfg, "enabled", None) or ["pullback"])
    missing = [n for n in names if n not in PLAYBOOK]
    if missing:
        raise UnknownStrategy(
            f"unknown strategy name(s): {missing}. Known: {sorted(PLAYBOOK)}")
    return [PLAYBOOK[n] for n in names]


def _row_signal(sym: str, idx: pd.Timestamp, row: pd.Series, prev: pd.Series,
                cfg) -> Optional[Signal]:
    """First enabled strategy that fires on this bar, or None.

    One trade per symbol per day, whichever rule set sees it first. Letting two
    strategies both open the same symbol would double the risk on one bet while
    every position limit still reported a single normal-sized trade.
    """
    for spec in enabled_specs(cfg):
        sig = spec.entry(sym, idx, row, prev, cfg)
        if sig is not None:
            return sig
    return None


def signals_for_symbol(sym: str, df: pd.DataFrame, cfg) -> List[Signal]:
    """Every historical signal for one symbol. Used by the backtester."""
    data = prepare(df, cfg)
    out: List[Signal] = []
    for i in range(1, len(data)):
        sig = _row_signal(sym, data.index[i], data.iloc[i], data.iloc[i - 1], cfg)
        if sig is not None:
            out.append(sig)
    return out


def latest_signal(sym: str, df: pd.DataFrame, cfg) -> Optional[Signal]:
    """Signal on the most recent completed bar only. Used by the live scanner."""
    data = prepare(df, cfg)
    if len(data) < 2:
        return None
    return _row_signal(sym, data.index[-1], data.iloc[-1], data.iloc[-2], cfg)


def exit_decision(df: pd.DataFrame, cfg, bars_held: Optional[int] = None,
                  strategy: str = "pullback") -> Optional[str]:
    """Should an open position be closed? Returns a reason, or None to hold.

    `strategy` decides which exit rules apply, because they are not
    interchangeable. Running the trend exit over a mean reversion trade would
    close it the day it was opened: that trade is entered below its short
    average on purpose.
    """
    data = prepare(df, cfg)
    if len(data) < 2:
        return None
    spec = PLAYBOOK.get(strategy) or PLAYBOOK["pullback"]
    return spec.exit(data.iloc[-1], cfg, bars_held)


def trend_state(df: pd.DataFrame, cfg) -> dict:
    """Human-readable snapshot of where a symbol stands right now."""
    data = prepare(df, cfg)
    row = data.iloc[-1]
    if pd.isna(row["sma_slow"]):
        return {"status": "warming up", "bars": len(data)}

    above_slow = row["close"] > row["sma_slow"]
    stacked = row["sma_fast"] > row["sma_slow"]
    dist_ema = (row["close"] - row["ema_pb"]) / row["ema_pb"] * 100

    if above_slow and stacked:
        status = "uptrend"
    elif not above_slow and row["sma_fast"] < row["sma_slow"]:
        status = "downtrend"
    else:
        status = "mixed"

    return {
        "status": status,
        "close": round(float(row["close"]), 2),
        "sma_fast": round(float(row["sma_fast"]), 2),
        "sma_slow": round(float(row["sma_slow"]), 2),
        "ema_pb": round(float(row["ema_pb"]), 2),
        "pct_from_ema": round(float(dist_ema), 2),
        "atr": round(float(row["atr"]), 2),
    }
