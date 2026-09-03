"""
The researcher.

Two layers, deliberately separate:

  1. A DETERMINISTIC earnings check. Earnings dates are a fact you can look
     up. You do not need a language model to read a calendar, and you should
     not use one for it, because a model can be wrong about a fact while
     sounding certain.

  2. An LLM review of recent news. This is the part that is actually AI, and
     it has exactly one power: it can VETO a trade the rules already chose.
     It cannot pick a trade, cannot increase size, cannot move a stop, cannot
     override a stop-out. That limit is enforced in code below, not by asking
     the model nicely.

Why so narrow: a language model cannot predict prices, and nothing in its
training makes it good at that. What it can do is read a headline and notice
"this company just guided down" or "there is a pending acquisition vote on
Tuesday", which is real information your moving averages cannot see. That is
a filter, and a filter is a job an LLM is genuinely good at.

If the research layer is unavailable, the agent SKIPS the trade rather than
taking it blind. Missing a trade costs you nothing but opportunity. Taking one
into an earnings report you did not check for costs money.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

# Cheap and fast, which is what a per-symbol yes/no check wants.
VETO_MODEL = os.getenv("RESEARCH_MODEL", "claude-haiku-4-5-20251001")
# One call a day, so a stronger model is affordable here.
BRIEF_MODEL = os.getenv("BRIEFING_MODEL", "claude-sonnet-5")


@dataclass
class Verdict:
    veto: bool
    reason: str
    flags: List[str]
    source: str            # "earnings", "llm", "unavailable", "none"
    headlines_seen: int = 0

    @property
    def allowed(self) -> bool:
        return not self.veto


# ---------------------------------------------------------------------------
# Layer 1: facts
# ---------------------------------------------------------------------------

# Funds do not report earnings. Asking Yahoo about them returns a 404 and a
# wall of red text that looks like a failure but is not one, so we skip the
# lookup entirely for anything known to be a fund.
FUNDS = {
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "VEA", "VWO", "EFA", "EEM",
    "AGG", "BND", "SHY", "IEF", "TLT", "LQD", "HYG", "BIL", "GLD", "SLV",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE",
    "XLC", "SMH", "SOXX", "ARKK", "IBIT", "RSP", "MDY", "IJH", "IJR",
}


def _quiet():
    """Silence yfinance's own chatter. It logs 404s for symbols that simply
    have no fundamentals, which is normal and not an error we can act on."""
    import logging
    for name in ("yfinance", "yfinance.data", "yfinance.ticker",
                 "yfinance.scrapers", "peewee"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def next_earnings_date(symbol: str) -> Optional[datetime]:
    """Next scheduled earnings date, or None if we cannot determine one."""
    if symbol.upper() in FUNDS:
        return None
    import contextlib
    import io

    try:
        import yfinance as yf
        _quiet()

        # Some yfinance versions print straight to stdout instead of logging,
        # so swallow both streams for the duration of the lookup.
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            t = yf.Ticker(symbol)

            try:
                df = t.get_earnings_dates(limit=8)
                if df is not None and len(df):
                    now = datetime.now(timezone.utc)
                    future = [d for d in df.index.to_pydatetime()
                              if d.replace(tzinfo=d.tzinfo or timezone.utc) > now]
                    if future:
                        d = min(future)
                        return d.replace(tzinfo=d.tzinfo or timezone.utc)
            except Exception:
                pass

            cal = getattr(t, "calendar", None)
            if isinstance(cal, dict):
                vals = cal.get("Earnings Date") or []
                if vals:
                    d = vals[0]
                    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
                    if dt > datetime.now(timezone.utc):
                        return dt
    except Exception:
        return None
    return None


def earnings_veto(symbol: str, within_days: int = 10) -> Verdict:
    """Refuse to open a swing trade that will be holding through earnings.

    An earnings report is a scheduled coin flip. The stock can gap 10% either
    way overnight, straight through your stop, and your position sizing
    assumed that could not happen. This is the single highest-value filter in
    the whole system and it needs no AI at all.
    """
    d = next_earnings_date(symbol)
    if d is None:
        return Verdict(False, "no earnings date found", [], "earnings")

    days = (d - datetime.now(timezone.utc)).days
    if 0 <= days <= within_days:
        return Verdict(
            True,
            f"earnings on {d.date()}, in {days} day(s)",
            ["earnings"],
            "earnings",
        )
    return Verdict(False, f"next earnings {d.date()}, {days} days out", [], "earnings")


def headlines(symbol: str, limit: int = 8, max_age_days: int = 14) -> List[dict]:
    """Recent news headlines. Titles and dates only, no article bodies."""
    out = []
    try:
        import yfinance as yf
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        for item in (yf.Ticker(symbol).news or [])[: limit * 2]:
            c = item.get("content", item)
            title = c.get("title") or ""
            if not title:
                continue
            pub = c.get("pubDate") or c.get("providerPublishTime")
            when = None
            if isinstance(pub, (int, float)):
                when = datetime.fromtimestamp(pub, tz=timezone.utc)
            elif isinstance(pub, str):
                try:
                    when = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                except ValueError:
                    when = None
            if when and when < cutoff:
                continue
            out.append({
                "title": title.strip()[:220],
                "publisher": (c.get("provider", {}) or {}).get("displayName", "")
                              or c.get("publisher", ""),
                "date": when.date().isoformat() if when else "",
            })
            if len(out) >= limit:
                break
    except Exception:
        return []
    return out


# ---------------------------------------------------------------------------
# Layer 2: the LLM
# ---------------------------------------------------------------------------

class Researcher:
    def __init__(self, api_key: Optional[str] = None, enabled: bool = True):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.enabled = enabled and bool(self.api_key)
        self.calls = 0
        self.errors = 0

    def _ask(self, model: str, system: str, user: str, max_tokens: int = 700) -> str:
        import requests

        resp = requests.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=45,
        )
        self.calls += 1
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
        blocks = resp.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    # -- the veto -----------------------------------------------------------

    VETO_SYSTEM = """You screen swing trades for event risk. You are a filter, not a forecaster.

You CANNOT predict prices and must never try. Do not reason about whether the
stock will go up or down. Do not comment on valuation, momentum, or technicals.

Veto a trade ONLY when the supplied headlines show a specific, dated, concrete
event that makes the next two weeks unusually hazardous. Examples that justify
a veto: a pending merger or acquisition vote, a regulatory decision with a
known date, a guidance cut or profit warning issued in the last few days, an
accounting investigation or restatement, a CEO or CFO departure announced this
week, a trading halt, a bankruptcy or going-concern filing, a product recall
or safety investigation.

Do NOT veto for: ordinary analyst upgrades or downgrades, price target
changes, routine product news, opinion or prediction articles, "is X a buy"
content, general market commentary, or an absence of news. Vague unease is not
a reason. When in doubt, allow the trade.

Reply with JSON only, no other text:
{"veto": true or false, "reason": "one short sentence", "flags": ["short-tag"]}"""

    def review_trade(self, symbol: str, entry: float, stop: float, target: float,
                     news: List[dict]) -> Verdict:
        """Ask the model whether a rule-generated trade should be blocked.

        The model's ONLY possible influence is turning a yes into a no. There
        is no code path anywhere that lets it create or enlarge a position.
        """
        if not self.enabled:
            return Verdict(False, "research disabled", [], "none", len(news))

        if not news:
            return Verdict(False, "no recent headlines found", [], "llm", 0)

        lines = "\n".join(f"- [{h['date']}] {h['title']} ({h['publisher']})" for h in news)
        user = (
            f"Ticker: {symbol}\n"
            f"Planned swing trade: buy near ${entry:.2f}, stop ${stop:.2f}, "
            f"target ${target:.2f}, expected hold about 5 to 15 trading days.\n\n"
            f"Headlines from the last 14 days:\n{lines}\n\n"
            f"Should this trade be vetoed for concrete event risk?"
        )

        try:
            raw = self._ask(VETO_MODEL, self.VETO_SYSTEM, user, max_tokens=300)
        except Exception as exc:
            self.errors += 1
            return Verdict(False, f"research call failed: {exc}", ["error"],
                           "unavailable", len(news))

        try:
            start, end = raw.find("{"), raw.rfind("}")
            parsed = json.loads(raw[start:end + 1])
        except Exception:
            self.errors += 1
            return Verdict(False, "could not parse research response", ["error"],
                           "unavailable", len(news))

        # Coerce hard. Anything unexpected means "do not veto", because a
        # malformed response must not silently become a trading decision.
        veto = parsed.get("veto") is True
        reason = str(parsed.get("reason", ""))[:240] or "no reason given"
        flags = [str(f)[:32] for f in (parsed.get("flags") or [])][:5]
        return Verdict(veto, reason, flags, "llm", len(news))

    # -- the briefing -------------------------------------------------------

    BRIEF_SYSTEM = """You write a short daily briefing for one retail swing trader.

Be plain and specific. No hype, no predictions, no price targets, no advice to
buy or sell. Describe what happened and what is scheduled, nothing more.

Never state or imply that any outcome is likely or guaranteed. If the news is
thin, say so in one line rather than padding.

Format: at most 200 words. Short paragraphs. No headers, no bullet lists."""

    def daily_briefing(self, positions: List[dict], signals: List[dict],
                       vetoes: List[dict], equity: float, day_change: float) -> str:
        if not self.enabled:
            return ("Research is off. Add ANTHROPIC_API_KEY to your .env file "
                    "to get a written briefing here.")

        pos = "\n".join(
            f"- {p['symbol']}: {p['shares']} sh @ ${p['avg_entry']:.2f}, "
            f"now worth ${p['market_value']:.2f}, open P&L ${p['unrealized_pl']:+.2f}"
            for p in positions) or "- none"
        sig = "\n".join(
            f"- {s['symbol']}: entry ~${s['entry']:.2f}, stop ${s['stop']:.2f}, "
            f"target ${s['target']:.2f}" for s in signals) or "- none"
        vet = "\n".join(f"- {v['symbol']}: {v['reason']}" for v in vetoes) or "- none"

        user = (
            f"Account equity ${equity:,.2f}, change today {day_change:+.2f}%.\n\n"
            f"Open positions:\n{pos}\n\n"
            f"New setups the rules found today:\n{sig}\n\n"
            f"Setups that were blocked, and why:\n{vet}\n\n"
            f"Write today's briefing."
        )
        try:
            return self._ask(BRIEF_MODEL, self.BRIEF_SYSTEM, user, max_tokens=600).strip()
        except Exception as exc:
            self.errors += 1
            return f"Briefing unavailable: {exc}"


# ---------------------------------------------------------------------------

def screen(symbol: str, entry: float, stop: float, target: float,
           researcher: Researcher, cfg_research) -> Verdict:
    """Run the full screen: facts first, then the model. Facts win.

    Order matters. If earnings are three days out, that is settled and there
    is no point spending a model call to have it agree.
    """
    if cfg_research.check_earnings:
        v = earnings_veto(symbol, cfg_research.earnings_blackout_days)
        if v.veto:
            return v

    if not cfg_research.use_llm:
        return Verdict(False, "LLM review disabled", [], "none")

    news = headlines(symbol, limit=cfg_research.max_headlines)
    v = researcher.review_trade(symbol, entry, stop, target, news)

    # Fail closed. If the research layer could not run, do not take the trade
    # blind. A missed trade costs nothing but opportunity.
    if v.source == "unavailable" and cfg_research.require_research:
        return Verdict(True, f"blocked: research unavailable ({v.reason})",
                       ["no-research"], "unavailable", v.headlines_seen)
    return v
