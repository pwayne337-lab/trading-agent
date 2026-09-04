"""
Indicators.

Every function here returns a series aligned to the input index, where the
value at row t uses only data from row t and earlier. That property is the
whole ballgame: if an indicator peeks at future bars, the backtest prints
beautiful numbers that you can never reproduce with real money.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average.

    adjust=False makes this the recursive form that charting platforms
    (TradingView, thinkorswim) actually draw, so the numbers here match what
    you see on screen.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: the largest of today's range, or either gap from yesterday."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average true range, Wilder's smoothing.

    ATR is a measure of how much a stock typically moves in a day. It is not
    a direction signal. We use it to size stops so that a $400 stock and a
    $40 stock get stops that are equally sensible relative to their own noise.
    """
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rolling_min(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).min()


def avg_dollar_volume(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """Average daily dollar volume, a proxy for how easily you can get filled."""
    return (close * volume).rolling(window=period, min_periods=period).mean()


def rolling_max(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).max()


def rsi(series: pd.Series, period: int = 2) -> pd.Series:
    """Relative strength index, Wilder smoothing.

    At a short period this is not a trend gauge, it is an "how stretched is
    this in the last couple of days" gauge, which is what the mean reversion
    rules want. A reading under 10 on a 2-period RSI means two straight days
    of almost nothing but selling.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # No losses at all in the window is a maximum reading, not a missing one.
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return out


def add_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Attach every indicator the strategy needs to an OHLCV frame.

    Expects columns: open, high, low, close, volume (lowercase).
    """
    out = df.copy()
    out["sma_fast"] = sma(out["close"], cfg.sma_fast)
    out["sma_slow"] = sma(out["close"], cfg.sma_slow)
    out["ema_pb"] = ema(out["close"], cfg.ema_pullback)
    out["atr"] = atr(out["high"], out["low"], out["close"], cfg.atr_period)
    out["swing_low"] = rolling_min(out["low"], cfg.swing_lookback)
    out["adv"] = avg_dollar_volume(out["close"], out["volume"], cfg.dollar_volume_window)

    # Used by the breakout rules. Shifted by one bar on purpose: the question
    # is whether today cleared the range that existed BEFORE today, and a
    # window that includes today's own high can never be cleared by it.
    out["donchian_hi"] = rolling_max(out["close"], cfg.breakout_lookback).shift(1)

    # Used by the mean reversion rules.
    out["rsi_fast"] = rsi(out["close"], cfg.reversion_rsi_period)
    out["sma_exit"] = sma(out["close"], cfg.reversion_sma)
    return out
