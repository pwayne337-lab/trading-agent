"""
Agent state.

Every run writes a JSON snapshot of what the agent saw and did, and appends
one row to an equity history file. The dashboard is built from these two
files and nothing else, which means the dashboard can never show you
something the agent did not actually record.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "agent_state.json"
EQUITY_FILE = STATE_DIR / "equity_history.csv"
RUNLOG_FILE = STATE_DIR / "run_log.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def blank_state() -> dict:
    return {
        "updated_at": None,
        "mode": "unknown",
        "healthy": False,
        "account": {},
        "positions": [],
        "signals": [],
        "vetoes": [],
        "orders": [],
        "exits": [],
        # Stops the agent placed behind positions it found unprotected.
        "protected": [],
        "skipped": [],
        "findings": [],
        # Trades this run declined to take on its own and handed to you
        # instead, and the moment they stop being valid. Empty is a real
        # answer: it means the run held nothing back.
        "pending": [],
        "pending_expires_at": None,
        # Anything config/overrides.json changed, clamped or refused, so a
        # setting you edited that did not take effect says so on the page.
        "config_notes": [],
        "briefing": "",
        "recent_trades": [],
        "errors": [],
        # Set when a run stopped before it read fresh numbers and the figures
        # above were carried over from the run named here. The dashboard dates
        # them by this stamp, so a page rebuilt by an aborted run cannot pass
        # off the previous run's snapshot as a new measurement.
        # When the trading logic itself last ran, as opposed to when the
        # figures on the page were last measured. The refresh job re-reads the
        # account every hour, so updated_at is almost always minutes old; if
        # that were the only timestamp, an agent that had stopped running
        # entirely would still look perfectly healthy.
        "last_full_run": None,
        "carried_from": None,
        "carried_reason": "",
        # Which rule set opened each open position. The broker does not record
        # this and cannot, but the exits differ per strategy, so without it a
        # mean reversion trade would be managed by the trend exit and closed
        # the day after it opened.
        "strategy_by_symbol": {},
        "research": {"llm_calls": 0, "llm_errors": 0, "enabled": False},
    }


def load_state() -> dict:
    if not STATE_FILE.exists():
        return blank_state()
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return blank_state()


def save_state(state: dict, full_run: bool = False) -> Path:
    """Write the state file. `full_run` means the trading logic ran this time,
    as opposed to a refresh or a monitoring check that only re-read the
    account, and it is what the dashboard's "last traded" marker is built on.
    """
    state["updated_at"] = now_iso()
    if full_run:
        state["last_full_run"] = state["updated_at"]
    elif not state.get("last_full_run"):
        # Never let a save ERASE the marker. _cmd_run builds its state from
        # blank_state() and carries almost nothing forward, so any path in it
        # that saves without full_run -- the market-hours guard, for one --
        # wrote a null over a real timestamp and made the page announce "the
        # agent has not traded" about an agent that traded yesterday.
        #
        # Held here rather than at each call site on purpose: this is the one
        # place every writer passes through, and the next command added to
        # this system should not have to know the rule exists.
        try:
            prior = json.loads(STATE_FILE.read_text()).get("last_full_run")
        except Exception:
            prior = None
        if prior:
            state["last_full_run"] = prior
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    return STATE_FILE


def append_equity(equity: float, cash: float, positions: int,
                  when: Optional[str] = None) -> Path:
    """One row per run. Duplicate dates are replaced, so re-running a day
    does not create two points on the chart."""
    stamp = (when or datetime.now(timezone.utc).date().isoformat())[:10]
    rows = []
    if EQUITY_FILE.exists():
        with EQUITY_FILE.open() as f:
            rows = [r for r in csv.DictReader(f) if r.get("date") != stamp]

    rows.append({
        "date": stamp,
        "equity": f"{equity:.2f}",
        "cash": f"{cash:.2f}",
        "positions": str(positions),
    })
    rows.sort(key=lambda r: r["date"])

    with EQUITY_FILE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "equity", "cash", "positions"])
        w.writeheader()
        w.writerows(rows)
    return EQUITY_FILE


def load_equity_history() -> list:
    if not EQUITY_FILE.exists():
        return []
    out = []
    with EQUITY_FILE.open() as f:
        for r in csv.DictReader(f):
            try:
                out.append({
                    "date": r["date"],
                    "equity": float(r["equity"]),
                    "cash": float(r["cash"]),
                    "positions": int(r["positions"]),
                })
            except (ValueError, KeyError):
                continue
    return out


def load_runs(limit: int = 20) -> list:
    """The most recent audit-trail entries, oldest first.

    Reading back what happened on previous runs is what lets a watcher tell one
    bad day from a pattern of them.
    """
    if not RUNLOG_FILE.exists():
        return []
    out = []
    try:
        lines = RUNLOG_FILE.read_text().strip().splitlines()[-limit:]
    except Exception:
        return []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def log_run(entry: dict) -> None:
    """Append-only audit trail. Never rewritten, so you can always reconstruct
    what the agent believed at the time it acted."""
    entry = {"at": now_iso(), **entry}
    with RUNLOG_FILE.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
