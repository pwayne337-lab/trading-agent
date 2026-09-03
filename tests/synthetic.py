"""Deterministic synthetic price series, used to validate the engine offline."""

import numpy as np
import pandas as pd


def make_series(n=1600, seed=7, drift=0.0004, vol=0.014, start=100.0,
                regime_flip_at=None):
    """Geometric random walk with intraday ranges, shaped like a real stock."""
    rng = np.random.default_rng(seed)
    mu = np.full(n, drift)
    if regime_flip_at is not None:
        mu[regime_flip_at:] = -abs(drift) * 1.2

    shocks = rng.normal(0, vol, n)
    closes = start * np.exp(np.cumsum(mu + shocks))

    # Build OHLC around each close so highs/lows are internally consistent.
    opens = np.empty(n)
    opens[0] = start
    gap = rng.normal(0, vol * 0.35, n)
    opens[1:] = closes[:-1] * (1 + gap[1:])

    span = np.abs(rng.normal(0, vol * 0.9, n)) * closes
    highs = np.maximum(opens, closes) + span * rng.uniform(0.2, 1.0, n)
    lows = np.minimum(opens, closes) - span * rng.uniform(0.2, 1.0, n)
    lows = np.maximum(lows, 0.01)

    volume = rng.integers(5_000_000, 40_000_000, n)

    idx = pd.bdate_range("2016-01-04", periods=n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    ).rename_axis("date")


def universe(symbols, seed0=1, **kw):
    return {s: make_series(seed=seed0 + i, **kw) for i, s in enumerate(symbols)}
