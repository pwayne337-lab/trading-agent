"""
Walk-forward parameter evolution, and the test of whether it is worth doing.

The idea people usually mean by "an evolving agent" is: periodically re-tune
the settings on recent history so the strategy keeps up with the market. That
is a real technique. Systematic funds use it. It is called walk-forward
optimization and it works like this:

    train on 2018-2020   ->   pick the best settings   ->   trade 2021 with them
    train on 2019-2021   ->   pick the best settings   ->   trade 2022 with them
    train on 2020-2022   ->   pick the best settings   ->   trade 2023 with them
                                                            ...and so on

The settings are never chosen using the period they are judged on. That single
property is what separates evolution from self-flattery.

What makes it dangerous is doing it without the second half. If you tune on
2018-2020 and then report how those settings did on 2018-2020, you have
learned nothing except that your search worked. Every strategy looks brilliant
on the data used to pick it.

So this module does not just evolve. It measures whether evolving beats
leaving the settings alone, out of sample, and it is entirely possible the
answer is no. That answer is worth more than a tuned parameter, because it
tells you where the ceiling actually is.

Three rules the search obeys:

  It only touches STRATEGY settings, never RISK settings. Reward-to-risk,
  hold time, which average to pull back to. Never position size, never the
  drawdown breaker, never the correlation cap. Those exist to keep you solvent
  and are not up for optimization by a machine that cannot be fired.

  It changes one thing at a time. A joint search over five parameters on this
  much data finds noise with near-certainty.

  Every candidate must clear the incumbent by a real margin on data it never
  saw, or the incumbent stays. Ties go to not changing anything.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import AgentConfig

# Only these may be tuned, and only inside these bounds. Anything that
# controls how much money is at stake is deliberately absent.
SEARCH_SPACE: Dict[str, List] = {
    "reward_risk": [1.5, 2.0, 2.5, 3.0, 4.0],
    "max_hold_days": [15, 25, 40, 60],
    "ema_pullback": [10, 15, 20, 30],
    "stop_atr_buffer": [0.0, 0.10, 0.25, 0.50],
    "swing_lookback": [3, 5, 8],
}

# A candidate has to beat the incumbent by at least this much expectancy,
# out of sample, to be worth the risk of changing anything.
MIN_EDGE_R = 0.02


@dataclass
class WindowResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    param: str
    chosen: object
    default: object
    chosen_test_R: float
    default_test_R: float
    chosen_test_n: int
    default_test_n: int


def _with(cfg: AgentConfig, param: str, value) -> AgentConfig:
    out = copy.deepcopy(cfg)
    setattr(out.strategy, param, value)
    return out


# The strategy cannot produce a signal until it has 200 bars behind it for the
# long moving average. So every window is handed extra history in front of the
# period being measured, and trades that start inside that warmup are then
# discarded. Without the warmup a one-year window trades only in its last few
# weeks; without discarding, the warmup period leaks into the score.
WARMUP = pd.Timedelta(days=420)

MIN_TRADES = 30


def _slice(bars: Dict[str, pd.DataFrame], start, end,
           warmup: pd.Timedelta = WARMUP) -> Dict[str, pd.DataFrame]:
    out = {}
    for s, d in bars.items():
        cut = d[(d.index >= start - warmup) & (d.index <= end)]
        if len(cut) > 220:
            out[s] = cut
    return out


def _score(bars, cfg, only_after=None) -> tuple:
    """Expectancy in R and trade count, counting only trades entered on or
    after `only_after` so the warmup history cannot flatter the result."""
    try:
        res = run_backtest(bars, cfg)
        df = res.trades_df()
        if only_after is not None and len(df):
            entered = pd.to_datetime(df["entry_date"])
            df = df[entered >= pd.Timestamp(only_after)]
        if len(df) < MIN_TRADES:
            return float("nan"), len(df)
        return float(df["R"].mean()), len(df)
    except Exception:
        return float("nan"), 0


def walk_forward(bars: Dict[str, pd.DataFrame], cfg: AgentConfig,
                 params: Optional[List[str]] = None,
                 n_windows: int = 4, train_years: float = 3.0,
                 test_years: float = 1.0,
                 progress=None) -> List[WindowResult]:
    """Tune on each training window, then judge on the window that follows."""
    params = params or list(SEARCH_SPACE.keys())

    all_dates = sorted(set().union(*[set(d.index) for d in bars.values()]))
    if not all_dates:
        return []
    first, last = pd.Timestamp(all_dates[0]), pd.Timestamp(all_dates[-1])

    # Enough history for the warmup, the training years and the judged years.
    span = WARMUP + pd.Timedelta(days=int(365.25 * (train_years + test_years)))
    if last - first < span:
        return []

    # Lay the windows out end to end, most recent last.
    step = pd.Timedelta(days=int(365.25 * test_years))
    train_len = pd.Timedelta(days=int(365.25 * train_years))

    results: List[WindowResult] = []
    for k in range(n_windows):
        test_end = last - step * (n_windows - 1 - k)
        test_start = test_end - step
        train_end = test_start - pd.Timedelta(days=1)
        train_start = train_end - train_len
        if train_start - WARMUP < first:
            if progress:
                progress(f"window {k+1}: skipped, not enough history before it")
            continue

        train = _slice(bars, train_start, train_end)
        test = _slice(bars, test_start, test_end)
        if not train or not test:
            if progress:
                progress(f"window {k+1}: skipped, not enough bars")
            continue

        base_test_R, base_test_n = _score(test, cfg, only_after=test_start)
        if base_test_R != base_test_R:
            if progress:
                progress(f"window {k+1}: skipped, only {base_test_n} trades in "
                         f"the judged year (need {MIN_TRADES})")
            continue

        for p in params:
            if progress:
                progress(f"window {k+1}/{n_windows}  tuning {p}")

            best_val, best_train_R = None, float("-inf")
            for v in SEARCH_SPACE[p]:
                r, n = _score(train, _with(cfg, p, v), only_after=train_start)
                if r == r and r > best_train_R:
                    best_train_R, best_val = r, v

            if best_val is None:
                continue

            chosen_R, chosen_n = _score(test, _with(cfg, p, best_val),
                                        only_after=test_start)
            results.append(WindowResult(
                train_start, train_end, test_start, test_end, p, best_val,
                getattr(cfg.strategy, p), chosen_R, base_test_R,
                chosen_n, base_test_n))

    return results


def verdict(results: List[WindowResult]) -> dict:
    """Did tuning actually beat leaving it alone, on data it never saw?"""
    if not results:
        return {"windows": 0, "note": "not enough history to walk forward"}

    rows = []
    for r in results:
        if r.chosen_test_R != r.chosen_test_R or r.default_test_R != r.default_test_R:
            continue
        rows.append({"param": r.param,
                     "chosen": r.chosen,
                     "default": r.default,
                     "diff": r.chosen_test_R - r.default_test_R,
                     "changed": r.chosen != r.default})
    if not rows:
        return {"windows": 0, "note": "no window produced enough trades to judge"}

    df = pd.DataFrame(rows)
    per_param = df.groupby("param").agg(
        tests=("diff", "size"),
        avg_gain_R=("diff", "mean"),
        times_better=("diff", lambda s: int((s > 0).sum())),
        times_it_changed=("changed", "sum"),
    ).reset_index().sort_values("avg_gain_R", ascending=False)

    overall = float(df["diff"].mean())
    wins = int((df["diff"] > 0).sum())

    return {
        "windows": len(df),
        "avg_out_of_sample_gain_R": round(overall, 4),
        "times_tuning_helped": wins,
        "times_tuning_hurt": len(df) - wins,
        "per_param": per_param,
        "worth_doing": bool(overall > MIN_EDGE_R and wins > len(df) * 0.6),
    }


def format_report(results: List[WindowResult], v: dict) -> str:
    lines = ["", "WALK-FORWARD EVOLUTION TEST", "=" * 74, ""]

    if not results:
        lines.append(f"  {v.get('note', 'nothing to report')}")
        return "\n".join(lines)

    lines.append("  Each row: settings picked on the training years, then judged")
    lines.append("  on the following year, which the search never saw.")
    lines.append("")
    lines.append(f"  {'Train':<24}{'Judged on':<24}{'Setting':<18}"
                 f"{'Picked':>8}{'vs base':>10}")
    lines.append("  " + "-" * 82)
    for r in results:
        if r.chosen_test_R != r.chosen_test_R:
            continue
        diff = r.chosen_test_R - r.default_test_R
        flag = "" if abs(diff) > 1e-9 else "  (same as default)"
        lines.append(
            f"  {str(r.train_start.date())+' to '+str(r.train_end.date()):<24}"
            f"{str(r.test_start.date())+' to '+str(r.test_end.date()):<24}"
            f"{r.param:<18}{str(r.chosen):>8}{diff:>+10.3f}{flag}")

    lines += ["", "  " + "-" * 82, ""]
    lines.append(f"  Tuning helped in {v['times_tuning_helped']} of "
                 f"{v['windows']} tests, hurt in {v['times_tuning_hurt']}.")
    lines.append(f"  Average out-of-sample gain from tuning: "
                 f"{v['avg_out_of_sample_gain_R']:+.4f}R per trade.")
    lines.append("")

    if v["worth_doing"]:
        lines.append("  VERDICT: tuning beat leaving the settings alone. Worth")
        lines.append("  running on a schedule, with every change logged.")
    else:
        lines.append("  VERDICT: tuning did NOT reliably beat leaving the settings")
        lines.append("  alone. On this much data the search is mostly finding noise,")
        lines.append("  and an agent that re-tuned itself here would be busy rather")
        lines.append("  than smart. Leave the settings fixed and spend the effort on")
        lines.append("  more symbols, which buys real trades instead of better guesses.")

    lines += ["", "  Per setting:", "  " + "-" * 74]
    for _, r in v["per_param"].iterrows():
        lines.append(f"    {r['param']:<20}{int(r['tests'])} tests   "
                     f"avg {r['avg_gain_R']:+.4f}R   "
                     f"better in {int(r['times_better'])}")
    lines.append("")
    return "\n".join(lines)
