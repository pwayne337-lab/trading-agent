"""
Choosing which setups to take when there are more than there are slots.

This is the most under-examined decision in most trading systems. The entry
rules get all the attention, but if the rules find 15 candidates and you can
hold 5, then something has to discard 10 of them, and whatever does that
discarding is now a bigger influence on your results than the entry rules are.

On this agent's own history, 62% of valid setups were discarded for want of a
slot. Until now the discarding was done by list order, which live meant
alphabetical: AAPL beat WMT because of how the alphabet goes. Worse, the
backtest used a different order from the live agent, so the backtest was not
predicting what the live agent would actually do.

One function decides it now, and both the backtest and the live run call it.

Every score uses only bars up to and including the signal bar. A ranking that
peeks at tomorrow would make the backtest look wonderful and be worthless.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

MODES = ("none", "momentum", "reward_risk", "liquidity")

# Momentum measured over roughly six months, skipping the most recent month.
# The skip is standard: stocks tend to reverse over the very short term, so
# momentum measured right up to today is polluted by that reversal.
MOM_LOOKBACK = 126
MOM_SKIP = 21


def score(mode: str, data: pd.DataFrame, i: int, stop: float) -> float:
    """Rank a candidate. Higher is better. Ties keep their original order.

    `data` must already have indicators attached, and `i` is the position of
    the signal bar. Nothing after `i` is read.
    """
    if mode == "none" or i < 1 or i >= len(data):
        return 0.0

    row = data.iloc[i]

    if mode == "momentum":
        j = i - MOM_SKIP
        k = j - MOM_LOOKBACK
        if k < 0:
            return float("-inf")     # not enough history to rank it fairly
        past = float(data["close"].iloc[k])
        recent = float(data["close"].iloc[j])
        if past <= 0:
            return float("-inf")
        return recent / past - 1.0

    if mode == "reward_risk":
        # Prefer a tight stop relative to the stock's own noise, because that
        # buys more shares for the same dollar of risk.
        atr = float(row.get("atr", float("nan")))
        close = float(row["close"])
        if not (atr == atr) or atr <= 0 or close <= stop:
            return float("-inf")
        return -((close - stop) / atr)

    if mode == "liquidity":
        adv = row.get("adv", float("nan"))
        return float(adv) if adv == adv else float("-inf")

    return 0.0


def order(candidates, mode: str):
    """Sort candidates best-first, stably, so ties keep their original order.

    `candidates` is a sequence of (key, score) pairs. Returns the keys.
    """
    if mode == "none":
        return [k for k, _ in candidates]
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda pair: (-_safe(pair[1][1]), pair[0]))
    return [c[0] for _, c in indexed]


def _safe(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("-inf")
    return f if f == f else float("-inf")
