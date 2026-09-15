"""
Settings the hub is allowed to change, and the bounds it cannot argue with.

config.py holds the defaults and the reasoning behind them. This file holds a
sparse overlay on top: only the values that have been moved off those defaults,
so `config/overrides.json` reads as a list of your decisions rather than a
second copy of the configuration.

The bounds below are the point of the module. A dashboard that can set
risk_per_trade to 0.9 is a dashboard that can empty the account in eleven
trades, and a form validating itself in the browser protects nothing -- the
file can be edited directly, and anything holding a write token can post
whatever it likes. So the limits live here, on the side that actually runs the
orders, and a value outside them is clamped loudly rather than honoured.

EDITABLE is also the single source of truth for the hub's settings form: the
agent emits it with `python agent.py schema`, the hub renders from that, and
the form therefore cannot offer a field the agent would refuse.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
OVERRIDES_FILE = ROOT / "config" / "overrides.json"


# ---------------------------------------------------------------------------
# What may be changed, and within what range
# ---------------------------------------------------------------------------
#
# Anything absent from this table cannot be set from outside, whatever the
# overrides file says. That is a deliberate allowlist rather than a blocklist:
# a new field added to config.py is not remotely editable until someone
# deliberately lists it here and picks its bounds.
#
# allow_live_trading is absent and must stay absent. It is the switch between
# fake money and real money, and no web form should ever be able to touch it.

EDITABLE: Dict[str, Dict[str, Any]] = {
    # --- risk -------------------------------------------------------------
    "risk.risk_per_trade": {
        "type": "float", "min": 0.001, "max": 0.02, "step": 0.0005,
        "label": "Risk per trade",
        "unit": "fraction of equity",
        "help": "A full stop-out costs this share of the account. 0.01 is 1%. "
                "Above 2% a normal losing streak becomes an account-ending one.",
    },
    "risk.max_open_positions": {
        "type": "int", "min": 1, "max": 30,
        "label": "Max open positions",
        "help": "Ceiling on concurrent trades. The correlation and risk caps "
                "are what actually bound risk; this bounds attention.",
    },
    "risk.max_position_pct": {
        "type": "float", "min": 0.05, "max": 0.50, "step": 0.01,
        "label": "Max position size",
        "unit": "fraction of equity",
        "help": "No single position may exceed this share of equity, however "
                "tight its stop.",
    },
    "risk.max_correlation": {
        "type": "float", "min": 0.50, "max": 0.99, "step": 0.01,
        "label": "Max correlation",
        "help": "Refuse a new trade moving this closely with something already "
                "held. Stops three tickers becoming one bet at triple size.",
    },
    "risk.max_gross_exposure": {
        "type": "float", "min": 0.10, "max": 1.00, "step": 0.05,
        "label": "Max gross exposure",
        "unit": "fraction of equity",
        "help": "Total cost basis of open positions. Capped at 1.00: this "
                "agent never uses margin.",
    },
    "risk.max_drawdown_halt": {
        "type": "float", "min": 0.05, "max": 0.50, "step": 0.01,
        "label": "Drawdown halt",
        "help": "Stop opening new trades once the account is this far below "
                "its high water mark.",
    },
    "risk.min_risk_fraction": {
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Smallest worthwhile trade",
        "help": "As a fraction of a normal risk unit. Below this the leftover "
                "room is better left unspent than turned into a one-share "
                "position that still costs a slot.",
    },

    # --- strategy ---------------------------------------------------------
    "strategy.enabled": {
        "type": "multi", "options": ["pullback", "breakout", "reversion"],
        "min_selected": 1,
        "label": "Rule sets",
        "help": "Which strategies may fire. They look for different things, so "
                "running all three raises trade count without doubling a bet.",
    },
    "strategy.rank_by": {
        "type": "enum",
        "options": ["none", "momentum", "reward_risk", "liquidity"],
        "label": "Rank candidates by",
        "help": "Which setups win when more qualify than there are free slots. "
                "Most setups get discarded for want of a slot, so this matters "
                "more than most entry rules.",
    },
    "strategy.reward_risk": {
        "type": "float", "min": 1.0, "max": 5.0, "step": 0.25,
        "label": "Reward:risk target",
        "help": "Target distance as a multiple of stop distance.",
    },
    "strategy.max_hold_days": {
        "type": "int", "min": 5, "max": 120,
        "label": "Max hold (days)",
        "help": "Close a trade that has gone nowhere for this long.",
    },
    "strategy.min_avg_dollar_volume": {
        "type": "float", "min": 1e6, "max": 1e9, "step": 1e6,
        "label": "Liquidity floor",
        "unit": "avg daily dollar volume",
        "help": "Thin names have wide spreads and gap hard.",
    },

    # --- approval gate ----------------------------------------------------
    "gate.enabled": {
        "type": "bool",
        "label": "Hold uncertain trades for approval",
        "help": "Off means the agent behaves exactly as it did before: clean "
                "trades submitted, vetoed trades discarded, nothing waits.",
    },
    "gate.hold_vetoed": {
        "type": "bool",
        "label": "Let me overrule a veto",
        "help": "A vetoed trade becomes a proposal instead of vanishing. It "
                "still does not happen unless you approve it.",
    },
    "gate.hold_flagged": {
        "type": "bool",
        "label": "Hold flagged trades",
        "help": "Trades the research layer noted something about but did not "
                "block. This is the setting that can cost you a trade you "
                "would otherwise have taken.",
    },
    "gate.hold_research_outage": {
        "type": "bool",
        "label": "Hold when research could not run",
    },
    "gate.hold_above_risk_fraction": {
        "type": "float", "min": 0.0, "max": 5.0, "step": 0.25,
        "label": "Hold trades risking more than",
        "unit": "x a normal risk unit",
        "help": "0 turns this off.",
    },

    # --- research ---------------------------------------------------------
    "research.check_earnings": {
        "type": "bool",
        "label": "Block trades near earnings",
        "help": "An earnings report is a scheduled overnight coin flip that "
                "can gap straight through a stop.",
    },
    "research.earnings_blackout_days": {
        "type": "int", "min": 0, "max": 30,
        "label": "Earnings blackout (days)",
    },
    "research.use_llm": {
        "type": "bool",
        "label": "AI headline review",
        "help": "The model can only ever block a trade. There is no path that "
                "lets it create or enlarge a position.",
    },
    "research.require_research": {
        "type": "bool",
        "label": "Skip the trade if research fails",
        "help": "On means a research outage costs you trades. Off means the "
                "agent trades without knowing whether earnings are two days "
                "out. On is the safe side.",
    },
}

# Fields the hub must never be able to set, listed so the intent is recorded
# rather than merely implied by absence.
FORBIDDEN = {
    "allow_live_trading": "the paper/live switch is not a dashboard control",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _coerce(spec: dict, value: Any) -> Tuple[Any, str]:
    """Return (accepted_value, note). note is '' when the value was taken as is."""
    kind = spec["type"]

    if kind == "bool":
        return bool(value), ""

    if kind in ("int", "float"):
        try:
            v = int(value) if kind == "int" else float(value)
        except (TypeError, ValueError):
            return None, f"{value!r} is not a number"
        lo, hi = spec["min"], spec["max"]
        if v < lo:
            return lo, f"raised {v} to the minimum {lo}"
        if v > hi:
            return hi, f"lowered {v} to the maximum {hi}"
        return v, ""

    if kind == "enum":
        v = str(value)
        if v not in spec["options"]:
            return None, f"{v!r} is not one of {', '.join(spec['options'])}"
        return v, ""

    if kind == "multi":
        if not isinstance(value, list):
            return None, "expected a list"
        chosen = [str(x) for x in value if str(x) in spec["options"]]
        dropped = [str(x) for x in value if str(x) not in spec["options"]]
        if len(chosen) < spec.get("min_selected", 0):
            return None, (f"needs at least {spec['min_selected']}; "
                          f"{chosen or 'nothing'} is not enough")
        note = f"ignored unknown {', '.join(dropped)}" if dropped else ""
        return chosen, note

    return None, f"unknown field type {kind}"


def load_raw() -> dict:
    """The overrides file as written, or {} when there is none."""
    if not OVERRIDES_FILE.exists():
        return {}
    try:
        data = json.loads(OVERRIDES_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        # A corrupt overrides file must not stop the agent trading on its
        # documented defaults. The note is surfaced on the dashboard.
        return {"_unreadable": True}


def apply(cfg, raw: dict = None) -> Tuple[Any, List[str]]:
    """Layer the overrides onto a config. Returns (config, notes).

    Notes are the human-readable record of anything that was clamped, ignored
    or refused. They go into the run's state so the dashboard can show that a
    setting you changed is not the setting in force.
    """
    raw = load_raw() if raw is None else raw
    notes: List[str] = []

    if raw.get("_unreadable"):
        return cfg, ["config/overrides.json could not be parsed; "
                     "running on the defaults in config.py"]
    if not raw:
        return cfg, notes

    sections = {"risk": cfg.risk, "strategy": cfg.strategy,
                "research": cfg.research, "costs": cfg.costs,
                "gate": cfg.gate}
    changes: Dict[str, Dict[str, Any]] = {k: {} for k in sections}

    for section, values in raw.items():
        if section.startswith("_") or section == "watchlist":
            continue
        if section in FORBIDDEN:
            notes.append(f"refused {section}: {FORBIDDEN[section]}")
            continue
        if section not in sections:
            notes.append(f"ignored unknown section {section!r}")
            continue
        if not isinstance(values, dict):
            notes.append(f"ignored {section!r}: expected an object")
            continue

        for field, value in values.items():
            key = f"{section}.{field}"
            if field in FORBIDDEN:
                notes.append(f"refused {key}: {FORBIDDEN[field]}")
                continue
            spec = EDITABLE.get(key)
            if spec is None:
                notes.append(f"refused {key}: not a remotely editable setting")
                continue
            accepted, note = _coerce(spec, value)
            if accepted is None:
                notes.append(f"refused {key}: {note}")
                continue
            if note:
                notes.append(f"{key}: {note}")
            changes[section][field] = accepted

    new = cfg
    for section, fields in changes.items():
        if fields:
            new = replace(new, **{section: replace(sections[section], **fields)})

    # Watchlist is add/remove rather than a whole list, so a 150-symbol file
    # does not have to be round-tripped through a browser to add one ticker,
    # and an update to DEFAULT_WATCHLIST still reaches you.
    wl = raw.get("watchlist") or {}
    if isinstance(wl, dict) and (wl.get("add") or wl.get("remove")):
        symbols = list(new.watchlist)
        remove = {str(s).upper() for s in (wl.get("remove") or [])}
        missing = remove - set(symbols)
        if missing:
            notes.append(f"watchlist: nothing to remove for {', '.join(sorted(missing))}")
        symbols = [s for s in symbols if s not in remove]
        for s in (wl.get("add") or []):
            s = str(s).upper().strip()
            if s and s not in symbols:
                symbols.append(s)
        new = replace(new, watchlist=symbols)

    return new, notes


def schema() -> dict:
    """What the hub renders its settings form from."""
    return {
        "fields": [{"key": k, **v} for k, v in EDITABLE.items()],
        "forbidden": FORBIDDEN,
        "watchlist": {"type": "symbols", "label": "Watchlist",
                      "help": "Add or remove tickers. Everything else stays "
                              "on the list in config.py."},
    }
