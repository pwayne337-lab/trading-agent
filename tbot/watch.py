"""
The watchers.

Three jobs that have nothing to do with predicting prices and everything to do
with noticing when the machine is quietly broken.

Automated systems rarely fail loudly. They fail by continuing to run while one
assumption stops being true: the price feed freezes on Friday's close, a stop
order silently fails to attach, the scheduled job stops firing and the last
good dashboard sits there looking current. Every one of those keeps producing
plausible output right up until it costs you money.

Unlike everything else in this agent, these checks are cheap to verify. A
prediction takes years to evaluate. "Is this position protected by a stop?"
has an answer right now, and it is either yes or no.

  check_data          is the price data real, fresh and sane
  check_broker        does the broker agree with what the agent believes
  check_run_health    is the agent actually running on schedule
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"


@dataclass
class Finding:
    severity: str
    agent: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def _biz_days_since(ts) -> Optional[int]:
    try:
        d = pd.Timestamp(ts).normalize()
    except Exception:
        return None
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    if d > today:
        return 0
    return int(np.busday_count(d.date(), today.date()))


# ---------------------------------------------------------------------------
# 1. Data quality
# ---------------------------------------------------------------------------

def check_data(bars: Dict[str, pd.DataFrame], expected: List[str],
               max_stale_days: int = 4, frozen_bars: int = 5) -> List[Finding]:
    """Is the price data real, current and internally sensible?"""
    out: List[Finding] = []

    missing = sorted(set(expected) - set(bars))
    if missing:
        sev = CRITICAL if len(missing) > len(expected) * 0.2 else WARNING
        out.append(Finding(sev, "data", f"{len(missing)} symbol(s) failed to "
                                        f"load: {', '.join(missing[:8])}"))

    stale, frozen, weird = [], [], []
    for sym, df in bars.items():
        if df is None or len(df) < 2:
            out.append(Finding(WARNING, "data", f"{sym}: almost no history"))
            continue

        age = _biz_days_since(df.index[-1])
        if age is not None and age > max_stale_days:
            stale.append(f"{sym} ({age}d)")

        # A feed that stops updating often repeats the last value rather than
        # returning nothing, which looks like a very calm stock.
        tail = df["close"].tail(frozen_bars)
        if len(tail) == frozen_bars and float(tail.std()) == 0.0:
            frozen.append(sym)

        # A move this large is usually a split or dividend adjustment that
        # landed wrong, not a real day. Worth a human glance either way.
        rets = df["close"].pct_change().tail(60).abs()
        if len(rets) and float(rets.max()) > 0.5:
            weird.append(f"{sym} ({float(rets.max())*100:.0f}%)")

    if stale:
        out.append(Finding(CRITICAL, "data",
                           f"stale prices, newest bar is old: {', '.join(stale[:8])}"))
    if frozen:
        out.append(Finding(CRITICAL, "data",
                           f"price frozen for {frozen_bars} bars, feed may be "
                           f"stuck: {', '.join(frozen[:8])}"))
    if weird:
        out.append(Finding(WARNING, "data",
                           f"a daily move over 50%, check for a bad split "
                           f"adjustment: {', '.join(weird[:6])}"))
    return out


# ---------------------------------------------------------------------------
# 2. Reconciliation
# ---------------------------------------------------------------------------

def _flatten_orders(open_orders) -> List[dict]:
    """Bracket legs arrive nested under the parent until it fills. Flatten so
    a leg is never missed just because of where the broker put it."""
    flat: List[dict] = []
    for o in (open_orders or []):
        if not isinstance(o, dict):
            continue
        flat.append(o)
        for leg in (o.get("legs") or []):
            if isinstance(leg, dict):
                flat.append(leg)
    return flat


def _stop_coverage(open_orders):
    """Shares covered by a working stop, per symbol, and every symbol with any
    sell order at all.

    Only a stop counts as protection. A take-profit limit sitting above the
    market is not an exit, it is a wish, and both legs of a bracket are sell
    orders. Counting sell orders instead of reading their type is how an
    unprotected position gets reported as safe.
    """
    stop_qty: Dict[str, int] = {}
    sells: Dict[str, int] = {}
    for o in _flatten_orders(open_orders):
        sym = o.get("symbol")
        if not sym or not str(o.get("side", "")).startswith("sell"):
            continue
        if str(o.get("status", "")).lower() in ("canceled", "cancelled", "filled",
                                                "expired", "rejected"):
            continue
        sells[sym] = sells.get(sym, 0) + 1
        if o.get("stop_price") in (None, ""):
            continue
        try:
            qty = abs(int(float(o.get("qty") or 0)))
        except (TypeError, ValueError):
            qty = 0
        stop_qty[sym] = stop_qty.get(sym, 0) + qty
    return stop_qty, sells


def unprotected(positions: List[dict], open_orders) -> Dict[str, int]:
    """Held symbols with no stop, or a stop too small, and how many shares are
    exposed. Same reading of the orders the watcher uses, so the report and the
    repair can never disagree about what counts as protected.
    """
    stop_qty, _ = _stop_coverage(open_orders)
    out: Dict[str, int] = {}
    for p in (positions or []):
        sym = p.get("symbol")
        if not sym:
            continue
        try:
            owned = abs(int(float(p.get("shares") or 0)))
        except (TypeError, ValueError):
            continue
        if owned < 1:
            continue
        gap = owned - stop_qty.get(sym, 0)
        if gap > 0:
            out[sym] = gap
    return out


def check_positions_have_data(positions: List[dict], bars: Dict[str, pd.DataFrame]
                              ) -> List[Finding]:
    """Every held position needs live bars or it cannot be managed.

    The daily exits read the latest close. A position whose data failed to load
    is skipped by that loop without a word, so a delisted or renamed ticker
    would sit in the account with nothing evaluating it while the agent goes on
    opening new trades.
    """
    blind = sorted(p["symbol"] for p in (positions or [])
                   if p.get("symbol")
                   and (bars.get(p["symbol"]) is None or len(bars.get(p["symbol"], [])) < 2))
    if not blind:
        return []
    return [Finding(CRITICAL, "data",
                    f"holding {', '.join(blind)} with no usable price data. "
                    f"The daily exits cannot run on these positions.")]


def check_broker(account: dict, positions: List[dict], open_orders: List[dict],
                 believed: Optional[List[dict]] = None) -> List[Finding]:
    """Does the broker's reality match what the agent thinks is true?

    The most important line in this function is the unprotected-position
    check. Every entry is submitted as a bracket so the stop goes on with the
    buy, but a leg can fail to attach, or be cancelled by hand, or be filled
    and leave its sibling orphaned. A position with no stop behind it is the
    one situation this whole system is designed to never be in, and nothing
    else in the agent would notice.
    """
    out: List[Finding] = []

    held = {p.get("symbol") for p in (positions or []) if p.get("symbol")}
    shares_held: Dict[str, int] = {}
    for p in (positions or []):
        if p.get("symbol") and p.get("shares") is not None:
            try:
                shares_held[p["symbol"]] = abs(int(float(p["shares"])))
            except (TypeError, ValueError):
                pass

    stop_qty, sells = _stop_coverage(open_orders)

    naked = sorted(s for s in held if s not in stop_qty)
    if naked:
        out.append(Finding(CRITICAL, "reconcile",
                           f"UNPROTECTED: no stop order behind "
                           f"{', '.join(naked)}. These positions have no exit "
                           f"working at the broker."))

    # A stop that covers part of the position is the failure a count of orders
    # cannot see. A partly filled entry, or a stop cancelled and replaced at
    # the wrong size, leaves shares exposed behind an order that looks present.
    short = []
    for sym, covered in stop_qty.items():
        owned = shares_held.get(sym)
        if owned and covered and covered < owned:
            short.append(f"{sym} ({covered} of {owned} sh)")
    if short:
        out.append(Finding(CRITICAL, "reconcile",
                           f"PARTLY UNPROTECTED: the stop covers fewer shares "
                           f"than are held: {', '.join(short)}"))

    orphan = sorted(s for s in sells if s not in held)
    if orphan:
        out.append(Finding(WARNING, "reconcile",
                           f"sell orders working with no position behind them: "
                           f"{', '.join(orphan)}"))

    if believed is not None:
        was = {p.get("symbol") for p in believed if p.get("symbol")}
        appeared = sorted(held - was)
        vanished = sorted(was - held)
        if appeared:
            out.append(Finding(INFO, "reconcile",
                               f"new since last run: {', '.join(appeared)}"))
        if vanished:
            out.append(Finding(INFO, "reconcile",
                               f"closed since last run: {', '.join(vanished)}"))

    equity = float(account.get("equity") or 0)
    if equity <= 0:
        out.append(Finding(CRITICAL, "reconcile", "account equity is zero or negative"))
    if account.get("trading_blocked"):
        out.append(Finding(CRITICAL, "reconcile", "the broker has blocked trading"))
    if float(account.get("buying_power") or 0) < 0:
        out.append(Finding(CRITICAL, "reconcile", "buying power is negative"))

    # Concentration is a risk rule, but a position that has grown past the cap
    # on its own is something only a watcher would catch.
    if equity > 0:
        for p in (positions or []):
            share = abs(float(p.get("market_value") or 0)) / equity
            if share > 0.35:
                out.append(Finding(WARNING, "reconcile",
                                   f"{p.get('symbol')} is {share*100:.0f}% of the "
                                   f"account, larger than any entry rule would allow"))
    return out


# ---------------------------------------------------------------------------
# 3. Run health
# ---------------------------------------------------------------------------

def check_run_health(previous_state: Optional[dict], history: List[dict],
                     max_gap_days: int = 3) -> List[Finding]:
    """Has the agent actually been running?"""
    out: List[Finding] = []

    if not previous_state or not previous_state.get("updated_at"):
        out.append(Finding(INFO, "watchdog", "first recorded run"))
        return out

    age = _biz_days_since(str(previous_state["updated_at"])[:10])
    if age is not None and age > max_gap_days:
        out.append(Finding(CRITICAL, "watchdog",
                           f"the previous run was {age} business days ago. The "
                           f"schedule may have stopped firing."))

    prev_errors = previous_state.get("errors") or []
    if prev_errors:
        out.append(Finding(WARNING, "watchdog",
                           f"last run reported {len(prev_errors)} error(s): "
                           f"{prev_errors[0]}"))

    if len(history) >= 2:
        dates = pd.to_datetime([h["date"] for h in history]).sort_values()
        gaps = np.busday_count(dates[:-1].date, dates[1:].date)
        big = int((gaps > max_gap_days).sum())
        if big:
            out.append(Finding(WARNING, "watchdog",
                               f"{big} gap(s) of more than {max_gap_days} business "
                               f"days in the recorded history"))
    return out


# ---------------------------------------------------------------------------

def run_all(bars, expected, account, positions, open_orders,
            previous_state, history) -> List[Finding]:
    findings: List[Finding] = []
    findings += check_run_health(previous_state, history)
    findings += check_data(bars, expected)
    findings += check_broker(account, positions, open_orders,
                             believed=(previous_state or {}).get("positions"))
    findings += check_positions_have_data(positions, bars)
    order = {CRITICAL: 0, WARNING: 1, INFO: 2}
    findings.sort(key=lambda f: order.get(f.severity, 3))
    return findings


def summarize(findings: List[Finding]) -> str:
    if not findings:
        return "All checks clean."
    n_c = sum(1 for f in findings if f.severity == CRITICAL)
    n_w = sum(1 for f in findings if f.severity == WARNING)
    bits = []
    if n_c:
        bits.append(f"{n_c} critical")
    if n_w:
        bits.append(f"{n_w} warning")
    return ", ".join(bits) if bits else "notes only"
