"""
The trade journal.

Every closed trade is recorded with the conditions that existed when it was
opened, then grouped so patterns can be measured instead of remembered.

Two things this deliberately does NOT do:

  It does not change the agent's behavior. Nothing here feeds back into
  sizing, entries or exits. An agent that retunes itself on its own recent
  results chases noise: it will find that "Tuesdays are good" after eleven
  Tuesdays and size up into the next one. You read the journal, you decide,
  you change the config yourself.

  It does not report a finding it cannot support. Every number comes with the
  sample size it was computed from, and anything below `MIN_SAMPLE` is printed
  as "not enough data" rather than as a result. This is the whole difference
  between a journal and a superstition generator.

The most useful number in here is `trades_needed`: given how noisy your
results actually are, how many trades it would take before an edge this size
could be told apart from luck. For most retail strategies the answer is
humbling, and it is better to know it than to keep drawing conclusions from
thirty trades.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Below this many trades in a bucket, we report the count and nothing else.
MIN_SAMPLE = 20

# Confidence multiplier. 1.96 is the usual 95% two-sided value.
Z = 1.96


def _stats(r: pd.Series) -> dict:
    """Expectancy in R for one group, with an honest error bar."""
    r = pd.to_numeric(r, errors="coerce").dropna()
    n = len(r)
    if n == 0:
        return {"n": 0}

    mean = float(r.mean())
    sd = float(r.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 and sd == sd else float("nan")

    out = {
        "n": n,
        "expectancy_R": round(mean, 3),
        "win_rate_pct": round(float((r > 0).mean() * 100), 1),
        "best_R": round(float(r.max()), 2),
        "worst_R": round(float(r.min()), 2),
    }

    if se == se and se > 0:
        lo, hi = mean - Z * se, mean + Z * se
        out["ci_low"] = round(lo, 3)
        out["ci_high"] = round(hi, 3)
        # An edge you cannot distinguish from zero is not yet an edge.
        out["real"] = bool(lo > 0 or hi < 0)
        # How many trades it would take for an effect this size to clear the
        # noise. n scales with (sd / effect) squared.
        if abs(mean) > 1e-9 and sd == sd:
            out["trades_needed"] = int(math.ceil((Z * sd / abs(mean)) ** 2))
    return out


def _bucket(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Break the journal down by one column and report each group honestly."""
    if col not in df.columns:
        return pd.DataFrame()

    rows = []
    for value, grp in df.groupby(col, dropna=True, observed=True):
        s = _stats(grp["R"])
        if s.get("n", 0) == 0:
            continue
        row = {col: value, "n": s["n"]}
        if s["n"] >= MIN_SAMPLE:
            row.update({
                "expectancy_R": s.get("expectancy_R"),
                "win_rate_pct": s.get("win_rate_pct"),
                "distinguishable_from_luck": s.get("real"),
            })
        else:
            row.update({"expectancy_R": None, "win_rate_pct": None,
                        "distinguishable_from_luck": None})
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values("n", ascending=False) if len(out) else out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the grouping columns the analysis wants, where the inputs exist."""
    out = df.copy()

    if "stop_atr" in out.columns and out["stop_atr"].notna().any():
        try:
            out["stop_width"] = pd.qcut(
                out["stop_atr"], 3, labels=["tight", "medium", "wide"],
                duplicates="drop")
        except (ValueError, TypeError):
            pass

    if "entry_date" in out.columns:
        d = pd.to_datetime(out["entry_date"], errors="coerce")
        out["weekday"] = d.dt.day_name()
        out["year"] = d.dt.year

    if "regime" in out.columns:
        out["regime"] = out["regime"].map(
            {True: "SPY above 200-day", False: "SPY below 200-day",
             "True": "SPY above 200-day", "False": "SPY below 200-day"}
        ).fillna(out["regime"])

    return out


def excursions(df: pd.DataFrame) -> dict:
    """Why winners won and losers lost, from how far each trade travelled.

    Two numbers per trade, both measured in R:

      worst_R  how far it went AGAINST you before it resolved
      best_R   how far it went FOR you before it resolved

    These answer questions the profit-and-loss column cannot.

      Losers whose best_R was large were winning trades you gave back. That
      points at the exit, not the entry.

      Winners whose worst_R was deep needed room to work. If that number sits
      near your stop distance, you are getting shaken out of trades that would
      have paid, and the fix is a wider stop and a smaller position, not a
      different setup.

      Losers whose worst_R is past -1 gapped through the stop. No entry rule
      prevents that. Only smaller size does.
    """
    if not {"worst_R", "best_R", "R"}.issubset(df.columns):
        return {}

    d = df.dropna(subset=["R"])
    win, lose = d[d["R"] > 0], d[d["R"] <= 0]
    out = {"n_win": len(win), "n_lose": len(lose)}

    if len(lose):
        out["loser_best_R"] = round(float(lose["best_R"].mean()), 2)
        out["loser_worst_R"] = round(float(lose["worst_R"].mean()), 2)
        out["losers_that_were_up_1R"] = round(float((lose["best_R"] >= 1.0).mean() * 100), 1)

        # Measured on the REALIZED loss, not on how far the bar's low reached.
        # A bar can trade well below your stop after filling you at the stop,
        # which costs you nothing extra. Only the fill price says what a trade
        # actually cost, so only the fill price can say you lost more than
        # you planned to.
        out["losers_past_planned"] = round(float((lose["R"] < -1.05).mean() * 100), 1)
        deep = lose[lose["R"] < -1.05]
        if len(deep):
            out["avg_overrun_R"] = round(float(deep["R"].mean()), 2)
    if len(win):
        out["winner_worst_R"] = round(float(win["worst_R"].mean()), 2)
        out["winner_best_R"] = round(float(win["best_R"].mean()), 2)
        out["winners_that_dipped_half_R"] = round(float((win["worst_R"] <= -0.5).mean() * 100), 1)
        out["winners_that_ran_past_target"] = round(float((win["best_R"] > 2.2).mean() * 100), 1)
    return out


def analyze(trades: pd.DataFrame) -> dict:
    """Full breakdown of a set of closed trades."""
    if trades is None or len(trades) == 0 or "R" not in trades.columns:
        return {"overall": {"n": 0}, "buckets": {}}

    df = add_derived(trades)
    overall = _stats(df["R"])

    buckets = {}
    for col, label in (("reason", "How the trade ended"),
                       ("regime", "Market regime at entry"),
                       ("stop_width", "How wide the stop was"),
                       ("symbol", "By symbol"),
                       ("weekday", "Day of week entered"),
                       ("year", "By year")):
        b = _bucket(df, col)
        if len(b):
            buckets[label] = b

    return {"overall": overall, "buckets": buckets, "trades": len(df),
            "excursions": excursions(df)}


def format_report(result: dict, max_rows: int = 12) -> str:
    """Plain-text journal report."""
    o = result.get("overall", {})
    n = o.get("n", 0)
    lines = ["", "TRADE JOURNAL", "=" * 72, ""]

    if n == 0:
        lines.append("  No closed trades recorded yet.")
        lines.append("  The journal fills in as positions close.")
        return "\n".join(lines)

    lines.append(f"  Closed trades         {n}")
    lines.append(f"  Expectancy            {o.get('expectancy_R', 0):+.3f}R per trade")
    lines.append(f"  Win rate              {o.get('win_rate_pct', 0):.1f}%")
    lines.append(f"  Best / worst          {o.get('best_R', 0):+.2f}R / {o.get('worst_R', 0):+.2f}R")

    if "ci_low" in o:
        lines.append(f"  95% range             {o['ci_low']:+.3f}R to {o['ci_high']:+.3f}R")
        if o.get("real"):
            lines.append("  Verdict               the edge clears the noise at this sample size")
        else:
            lines.append("  Verdict               NOT distinguishable from luck yet")

    if "trades_needed" in o:
        need = o["trades_needed"]
        lines.append(f"  Trades needed         about {need:,} to call an edge this "
                     f"size real")
        if need > n:
            years = need / 100.0     # roughly 100 trades a year at 2 a week
            lines.append(f"                        you have {n}. At two trades a week "
                          f"that is ~{years:.0f} more years.")

    ex = result.get("excursions") or {}
    if ex:
        lines += ["", "  WHY THEY WON AND LOST", "  " + "-" * 68]
        if ex.get("n_lose"):
            lines.append(f"    Losing trades ({ex['n_lose']})")
            lines.append(f"      went {ex['loser_best_R']:+.2f}R in profit on average "
                         f"before turning around")
            lines.append(f"      {ex['losers_that_were_up_1R']:.0f}% were up a full 1R "
                         f"at some point and still lost")
            over = ex.get("avg_overrun_R")
            tail = f", averaging {over:+.2f}R when it happens" if over else ""
            lines.append(f"      {ex['losers_past_planned']:.0f}% cost MORE than the "
                         f"1R you planned to risk{tail}")
        if ex.get("n_win"):
            lines.append(f"    Winning trades ({ex['n_win']})")
            lines.append(f"      went {ex['winner_worst_R']:+.2f}R against you on average "
                         f"before working")
            lines.append(f"      {ex['winners_that_dipped_half_R']:.0f}% dipped at least "
                         f"half a stop first")
            lines.append(f"      {ex['winners_that_ran_past_target']:.0f}% kept running "
                         f"well past the target after you sold")

    for label, table in result.get("buckets", {}).items():
        lines += ["", f"  {label}", "  " + "-" * 68]
        head = table.head(max_rows)
        for _, r in head.iterrows():
            key = str(r.iloc[0])[:22]
            # pandas turns a None into NaN once it is in a column, so the
            # "not enough data" marker has to be tested for both.
            if pd.isna(r.get("expectancy_R")):
                lines.append(f"    {key:<24}{int(r['n']):>5} trades   "
                             f"not enough data (need {MIN_SAMPLE})")
            else:
                mark = "  <- clears the noise" if r.get("distinguishable_from_luck") else ""
                lines.append(f"    {key:<24}{int(r['n']):>5} trades   "
                             f"{r['expectancy_R']:+.3f}R   "
                             f"{r['win_rate_pct']:.0f}% win{mark}")
        if len(table) > max_rows:
            lines.append(f"    ... and {len(table) - max_rows} more")

    lines += [
        "",
        "  " + "-" * 68,
        "  Nothing here changes what the agent does. It is a record for you to",
        "  read. A bucket that looks good on 15 trades is noise, which is why",
        "  those rows refuse to show a number at all.",
        "",
    ]
    return "\n".join(lines)
