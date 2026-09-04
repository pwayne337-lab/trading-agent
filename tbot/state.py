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
        "skipped": [],
        "findings": [],
        "briefing": "",
        "recent_trades": [],
        "errors": [],
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


def save_state(state: dict) -> Path:
    state["updated_at"] = now_iso()
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


def log_run(entry: dict) -> None:
    """Append-only audit trail. Never rewritten, so you can always reconstruct
    what the agent believed at the time it acted."""
    entry = {"at": now_iso(), **entry}
    with RUNLOG_FILE.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
