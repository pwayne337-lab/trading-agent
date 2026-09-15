"""
Trades the agent decided not to take on its own.

The agent stays autonomous. A setup that passes sizing, the correlation check
and the research screen cleanly is submitted exactly as before -- the gate has
no opinion about it and adds no latency. What changes is the handling of the
trades the agent was not sure about, which until now were discarded with a line
in the log: they become proposals that wait for you, and expire if you do not
answer.

Why waiting is free here. submit_bracket sends a market order with
time_in_force "gtc", after the close, which fills at the next session's open.
An approval given overnight and submitted before that open fills at the same
open on the same terms. The gate inserts a person into the loop without moving
the fill.

Why they expire. A proposal is priced off one specific signal bar. Approving a
two-day-old one would fire a market order against a stop and target computed
from a session that has since been overtaken. Anything not answered by the open
it was written for is dead, and the morning job refuses it.

Two files, two owners:
  state/pending.json    the agent writes, the hub reads
  state/decisions.json  the hub writes, the agent reads
Neither side writes the other's file, so a hub edit landing while a run is in
flight cannot corrupt the run's own record.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import state as _state

# Resolved on every call rather than captured at import.
#
# state.py's paths are module attributes that the smoke test rebinds to a
# temporary directory so an end-to-end run cannot overwrite the real record.
# Caching a copy of them here defeated that: the test wrote its synthetic
# proposals straight into state/, and protect-now.yml -- which runs the tests
# and then commits `git add -A state` -- would have published a fake pending
# queue over the real one, discarding trades that were waiting on an answer.
#
# Reading through state.py keeps one place to redirect, and anything added to
# this module later inherits that for free.

def _dir() -> Path:
    return Path(_state.STATE_DIR)


def pending_file() -> Path:
    return _dir() / "pending.json"


def decisions_file() -> Path:
    return _dir() / "decisions.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def proposal_id(session: str, symbol: str, strategy: str) -> str:
    """Stable across re-runs of the same session.

    A run repeated by hand after a failure must not produce a second proposal
    for a trade you have already answered, and must not lose the answer.
    """
    return f"{session}-{symbol}-{strategy}"


# ---------------------------------------------------------------------------
# What counts as flagged
# ---------------------------------------------------------------------------

def flag_reasons(verdict, order, equity: float, cfg) -> List[str]:
    """Why this trade should wait for a person. Empty means take it normally.

    Every condition here is built on a signal the agent already produces. None
    of it is a new opinion about the trade: the research layer's own veto, its
    own flags, its own failure to run, and the size the risk module already
    computed.
    """
    gate = cfg.gate
    reasons: List[str] = []
    if not gate.enabled:
        return reasons

    if verdict.veto:
        if not gate.hold_vetoed:
            return []          # vetoed and not held: discarded, as before
        reasons.append(f"research vetoed it — {verdict.reason}")
    elif verdict.flags and gate.hold_flagged:
        tags = ", ".join(verdict.flags)
        reasons.append(f"research flagged [{tags}] — {verdict.reason}")

    if gate.hold_research_outage and verdict.source == "unavailable":
        reasons.append("the research layer could not run, so this trade was "
                       "never actually screened")

    # A trade risking far more than a normal unit is worth a glance even when
    # nothing else is wrong with it.
    if gate.hold_above_risk_fraction > 0:
        normal = equity * cfg.risk.risk_per_trade
        if normal > 0 and order.dollars_at_risk > normal * gate.hold_above_risk_fraction:
            reasons.append(
                f"risks ${order.dollars_at_risk:,.2f}, more than "
                f"{gate.hold_above_risk_fraction:g}x the usual "
                f"${normal:,.2f}")

    return reasons


# ---------------------------------------------------------------------------
# Writing and reading
# ---------------------------------------------------------------------------

def build_proposal(session: str, symbol: str, sig, order, verdict,
                   reasons: List[str]) -> dict:
    return {
        "id": proposal_id(session, symbol, sig.strategy),
        "kind": "trade",
        "symbol": symbol,
        "strategy": sig.strategy,
        "shares": order.shares,
        "entry": round(order.entry, 2),
        "stop": round(order.stop, 2),
        "target": round(order.target, 2),
        "dollars_at_risk": round(order.dollars_at_risk, 2),
        "notional": round(order.notional, 2),
        "why": sig.notes,
        "held_because": reasons,
        "research": {
            "veto": bool(verdict.veto),
            "reason": verdict.reason,
            "flags": list(verdict.flags or []),
            "source": verdict.source,
            "headlines_seen": getattr(verdict, "headlines_seen", 0),
        },
    }


def save_pending(proposals: List[dict], session: str,
                 expires_at: Optional[str], notes: List[str] = None) -> dict:
    """Replace the pending file with this run's proposals.

    Wholesale replacement is deliberate. Proposals belong to one session, and
    merging yesterday's into today's is how a stale trade gets submitted.
    """
    _dir().mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_at": _now(),
        "session": session,
        "expires_at": expires_at,
        "proposals": proposals,
        "notes": notes or [],
    }
    pending_file().write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def load_pending() -> dict:
    if not pending_file().exists():
        return {"proposals": [], "session": None, "expires_at": None}
    try:
        doc = json.loads(pending_file().read_text())
    except Exception:
        return {"proposals": [], "session": None, "expires_at": None,
                "error": "pending.json could not be parsed"}
    doc.setdefault("proposals", [])
    return doc


def load_decisions() -> Dict[str, dict]:
    """Human calls, keyed by proposal id.

    Keyed rather than a list so that answering twice is idempotent and the hub
    never has to read the file to append to it correctly.
    """
    if not decisions_file().exists():
        return {}
    try:
        doc = json.loads(decisions_file().read_text())
    except Exception:
        return {}
    got = doc.get("decisions", doc)
    return got if isinstance(got, dict) else {}


def is_expired(doc: dict, now: datetime = None) -> bool:
    exp = doc.get("expires_at")
    if not exp:
        return False
    try:
        t = datetime.fromisoformat(str(exp))
    except ValueError:
        return False
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (now or datetime.now(timezone.utc)) >= t


def resolve(pending: dict = None, decisions: Dict[str, dict] = None,
            now: datetime = None) -> dict:
    """Sort this session's proposals into what the morning job should do.

    `approved` is the only bucket that becomes an order, and only when the
    batch has not expired. Everything else is reported and dropped.
    """
    pending = load_pending() if pending is None else pending
    decisions = load_decisions() if decisions is None else decisions
    expired = is_expired(pending, now)

    out = {"approved": [], "rejected": [], "undecided": [],
           "expired": [], "session": pending.get("session"),
           "expires_at": pending.get("expires_at"), "batch_expired": expired}

    for p in pending.get("proposals", []):
        d = decisions.get(p["id"])
        verdict = (d or {}).get("verdict")
        item = {**p, "decision": d}
        if expired:
            out["expired"].append(item)
        elif verdict == "approve":
            out["approved"].append(item)
        elif verdict == "reject":
            out["rejected"].append(item)
        else:
            out["undecided"].append(item)
    return out


def summarise(res: dict) -> str:
    bits = [f"{len(res[k])} {k}" for k in
            ("approved", "rejected", "undecided", "expired") if res[k]]
    return ", ".join(bits) if bits else "nothing pending"
