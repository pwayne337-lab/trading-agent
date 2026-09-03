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
from typing import List, Optional

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

    def target_from(self, entry: float, reward_risk: float) -> float:
        return entry + reward_risk * (entry - self.stop)

    def risk_per_share_from(self, entry: float) -> float:
        return entry - self.stop


def _row_signal(sym: str, idx: pd.Timestamp, row: pd.Series, prev: pd.Series,
                cfg) -> Optional[Signal]:
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
        notes=f"stop {stop_atr:.2f}x ATR below entry",
    )


def prepare(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Attach indicators to a raw OHLCV frame."""
    return add_indicators(df, cfg)


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


def exit_decision(df: pd.DataFrame, cfg, bars_held: Optional[int] = None
                  ) -> Optional[str]:
    """Should an open position be closed? Returns a reason, or None to hold.

    The stop and the target look after themselves: they are submitted to the
    broker as bracket legs when the trade opens, so they fire the moment price
    reaches them, at any hour, whether or not this program is running.

    These two exits are different. They depend on where the CLOSE lands
    relative to a moving average, or on how long the trade has been open, so
    nothing at the broker can evaluate them. Somebody has to look once a day
    and act. That somebody is this function.

    In the backtest, these two accounted for more than half of all exits, so a
    live agent that skips them is not running the strategy that was tested.
    """
    data = prepare(df, cfg)
    if len(data) < 2:
        return None
    row = data.iloc[-1]

    if cfg.exit_on_trend_break and not pd.isna(row["sma_fast"]):
        if float(row["close"]) < float(row["sma_fast"]):
            return (f"trend break: closed at ${float(row['close']):,.2f}, "
                    f"below the {cfg.sma_fast}-day average of "
                    f"${float(row['sma_fast']):,.2f}")

    if bars_held is not None and bars_held >= cfg.max_hold_days:
        return (f"time stop: held {bars_held} sessions, limit is "
                f"{cfg.max_hold_days}")

    return None


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
