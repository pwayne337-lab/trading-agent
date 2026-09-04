"""
Broker adapter (Alpaca).

Why Alpaca and not Robinhood: Robinhood has no official API for stocks or
options. Its only public API covers crypto. People automate Robinhood stock
orders with unofficial libraries that impersonate the mobile app, which
violates Robinhood's terms of service and has gotten accounts restricted.
Do not put your brokerage account at risk to save yourself a signup form.

Alpaca gives you a free paper account with a real API and the same code path
as live trading, so you can run this for months without risking a dollar.

Three separate locks stand between this code and a live order:
  1. AgentConfig.allow_live_trading must be True
  2. ALPACA_BASE_URL must point at the live endpoint
  3. --i-understand-the-risk must be passed on the command line
All three, every time. There is no "remember my choice".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class BrokerError(RuntimeError):
    pass


@dataclass
class Fill:
    symbol: str
    shares: int
    order_id: str
    status: str
    submitted: bool
    detail: str = ""


class AlpacaBroker:
    def __init__(self, key: Optional[str] = None, secret: Optional[str] = None,
                 base_url: Optional[str] = None, dry_run: bool = True):
        self.key = key or os.getenv("ALPACA_API_KEY", "")
        self.secret = secret or os.getenv("ALPACA_API_SECRET", "")
        self.base_url = (base_url or os.getenv("ALPACA_BASE_URL", PAPER_URL)).rstrip("/")
        self.dry_run = dry_run

    # -- properties ---------------------------------------------------------

    @property
    def is_live(self) -> bool:
        return self.base_url.startswith(LIVE_URL)

    @property
    def configured(self) -> bool:
        return bool(self.key and self.secret)

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs):
        try:
            import requests
        except ImportError as exc:
            raise BrokerError("requests is not installed. pip install -r requirements.txt") from exc

        if not self.configured:
            raise BrokerError(
                "no Alpaca credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET "
                "in your environment or in a .env file."
            )

        url = f"{self.base_url}{path}"
        try:
            resp = requests.request(method, url, headers=self._headers(),
                                    timeout=20, **kwargs)
        except requests.RequestException as exc:
            # A dropped connection, a DNS failure, a timeout, a proxy refusing
            # the tunnel. Every caller already handles BrokerError and falls
            # back to something sensible; none of them handle a raw urllib
            # exception, which would end the daily run in a stack trace with no
            # state saved and no dashboard written.
            raise BrokerError(f"{method} {path} failed to reach the broker: {exc}") from exc
        if resp.status_code >= 400:
            raise BrokerError(f"{method} {path} -> {resp.status_code}: {resp.text[:400]}")
        return resp.json() if resp.text else {}

    # -- reads --------------------------------------------------------------

    def account(self) -> dict:
        a = self._request("GET", "/v2/account")
        return {
            "equity": float(a.get("equity", 0)),
            "cash": float(a.get("cash", 0)),
            "buying_power": float(a.get("buying_power", 0)),
            "status": a.get("status"),
            "pattern_day_trader": a.get("pattern_day_trader"),
            "trading_blocked": a.get("trading_blocked"),
            "mode": "LIVE" if self.is_live else "PAPER",
        }

    def positions(self) -> List[dict]:
        return [
            {
                "symbol": p["symbol"],
                "shares": int(float(p["qty"])),
                "avg_entry": float(p["avg_entry_price"]),
                "market_value": float(p["market_value"]),
                "unrealized_pl": float(p["unrealized_pl"]),
            }
            for p in self._request("GET", "/v2/positions")
        ]

    def open_orders(self) -> List[dict]:
        return self._request("GET", "/v2/orders", params={"status": "open"})

    def clock(self) -> dict:
        return self._request("GET", "/v2/clock")

    def activities(self, page_size: int = 100) -> List[dict]:
        """Raw fill records, newest first."""
        return self._request("GET", "/v2/account/activities",
                             params={"activity_types": "FILL", "page_size": page_size})

    def realized_trades(self, limit: int = 20) -> List[dict]:
        """Closed round trips with realized P&L, newest first.

        Alpaca reports fills, not trades, so buys and sells are matched here
        first in, first out. A partial fill that closes half a position shows
        up as its own row, which is correct: that half is realized.
        """
        fills = sorted(self.activities(), key=lambda a: a.get("transaction_time", ""))
        lots: Dict[str, list] = {}
        closed: List[dict] = []

        for f in fills:
            sym = f.get("symbol")
            side = f.get("side", "")
            try:
                qty = abs(int(float(f.get("qty", 0))))
                price = float(f.get("price", 0))
            except (TypeError, ValueError):
                continue
            if not sym or qty <= 0:
                continue

            if side.startswith("buy"):
                lots.setdefault(sym, []).append([qty, price])
                continue

            remaining, cost, matched = qty, 0.0, 0
            queue = lots.get(sym, [])
            while remaining > 0 and queue:
                lot_qty, lot_price = queue[0]
                take = min(lot_qty, remaining)
                cost += take * lot_price
                matched += take
                remaining -= take
                lot_qty -= take
                if lot_qty == 0:
                    queue.pop(0)
                else:
                    queue[0][0] = lot_qty
            if matched == 0:
                continue

            proceeds = matched * price
            closed.append({
                "symbol": sym,
                "closed": (f.get("transaction_time") or "")[:10],
                "shares": matched,
                "avg_cost": round(cost / matched, 2),
                "exit": round(price, 2),
                "pnl": round(proceeds - cost, 2),
                "reason": "sold",
            })

        closed.reverse()
        return closed[:limit]

    # -- writes -------------------------------------------------------------

    def submit_bracket(self, symbol: str, shares: int, stop: float, target: float,
                       allow_live: bool = False, acknowledged: bool = False) -> Fill:
        """Submit a market buy wrapped in a bracket: a stop loss and a take
        profit that are attached to the position from the moment it fills.

        The bracket is the point. A bare market buy with the intention of
        setting a stop later is how people lose more than they planned. The
        exit orders go in with the entry, in the same request.
        """
        if self.is_live:
            if not allow_live:
                raise BrokerError(
                    "refusing to trade a LIVE account: allow_live_trading is False in config"
                )
            if not acknowledged:
                raise BrokerError(
                    "refusing to trade a LIVE account: pass --i-understand-the-risk"
                )

        if shares < 1:
            return Fill(symbol, 0, "", "rejected", False, "share count below 1")

        payload = {
            "symbol": symbol,
            "qty": str(shares),
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "order_class": "bracket",
            "take_profit": {"limit_price": round(target, 2)},
            "stop_loss": {"stop_price": round(stop, 2)},
        }

        if self.dry_run:
            return Fill(symbol, shares, "", "dry-run", False,
                        f"would buy {shares} {symbol}, stop {stop:.2f}, target {target:.2f}")

        order = self._request("POST", "/v2/orders", json=payload)
        return Fill(symbol, shares, order.get("id", ""), order.get("status", "?"), True)

    def submit_stop(self, symbol: str, shares: int, stop: float,
                    allow_live: bool = False, acknowledged: bool = False) -> Fill:
        """Put a plain protective stop behind a position that has none.

        Good til cancelled and on its own, not as a bracket leg, because there
        is no entry to attach it to. A stop is also one of the few orders a
        broker will accept outside market hours: it rests until price reaches
        it, so a position found unprotected in the evening can be covered that
        evening instead of spending the night naked.
        """
        if self.is_live:
            if not allow_live:
                raise BrokerError(
                    "refusing to trade a LIVE account: allow_live_trading is False in config")
            if not acknowledged:
                raise BrokerError(
                    "refusing to trade a LIVE account: pass --i-understand-the-risk")

        if shares < 1:
            return Fill(symbol, 0, "", "rejected", False, "share count below 1")
        if stop <= 0:
            return Fill(symbol, 0, "", "rejected", False, "stop price is not positive")

        if self.dry_run:
            return Fill(symbol, shares, "", "dry-run", False,
                        f"would protect {shares} {symbol} with a stop at {stop:.2f}")

        order = self._request("POST", "/v2/orders", json={
            "symbol": symbol,
            "qty": str(shares),
            "side": "sell",
            "type": "stop",
            "stop_price": round(stop, 2),
            "time_in_force": "gtc",
        })
        return Fill(symbol, shares, order.get("id", ""), order.get("status", "?"), True)

    def cancel_orders_for(self, symbol: str) -> List[dict]:
        """Cancel every working order on one symbol. Returns what was cancelled.

        Needed before closing a position: the bracket's stop and target legs
        are themselves orders that reserve the shares, so a close attempt
        while they are alive is rejected for insufficient quantity.

        The cancelled orders come back rather than a count because between the
        cancel and the close the position is standing there with no exit behind
        it. If the close then fails, this is the only record of what the stop
        used to be, and it is what makes putting it back possible.
        """
        if self.dry_run:
            return []
        killed: List[dict] = []
        for o in self.open_orders():
            if o.get("symbol") == symbol and o.get("id"):
                try:
                    self._request("DELETE", f"/v2/orders/{o['id']}")
                    killed.append(o)
                except BrokerError:
                    pass   # already filled or cancelled between list and delete
        return killed

    def _restore_stop(self, symbol: str, cancelled: List[dict]) -> bool:
        """Put a protective stop back after a close attempt failed.

        Best effort by design. It runs in the one situation the whole system
        exists to avoid, so it tries the simplest order that can work and
        reports honestly whether it got one in.
        """
        stop_price, qty = None, 0
        for o in cancelled:
            if not str(o.get("side", "")).startswith("sell"):
                continue
            raw = o.get("stop_price")
            if raw in (None, ""):
                continue
            try:
                stop_price = float(raw)
                qty = abs(int(float(o.get("qty") or 0)))
            except (TypeError, ValueError):
                continue
            break

        if stop_price is None or qty < 1:
            return False
        try:
            self._request("POST", "/v2/orders", json={
                "symbol": symbol,
                "qty": str(qty),
                "side": "sell",
                "type": "stop",
                "stop_price": round(stop_price, 2),
                "time_in_force": "gtc",
            })
            return True
        except BrokerError:
            return False

    def entry_dates(self) -> Dict[str, str]:
        """Most recent buy-fill date per symbol, for the time stop."""
        out: Dict[str, str] = {}
        try:
            fills = sorted(self.activities(), key=lambda a: a.get("transaction_time", ""))
        except BrokerError:
            return out
        for f in fills:
            sym, side = f.get("symbol"), f.get("side", "")
            when = (f.get("transaction_time") or "")[:10]
            if not sym or not when:
                continue
            if side.startswith("buy"):
                out[sym] = when
            elif side.startswith("sell"):
                out.pop(sym, None)     # position closed, clock resets
        return out

    def close_position(self, symbol: str, allow_live: bool = False,
                       acknowledged: bool = False) -> Fill:
        if self.is_live and not (allow_live and acknowledged):
            raise BrokerError("refusing to close a LIVE position without both safety flags")
        if self.dry_run:
            return Fill(symbol, 0, "", "dry-run", False, f"would close {symbol}")

        cancelled = self.cancel_orders_for(symbol)
        try:
            order = self._request("DELETE", f"/v2/positions/{symbol}")
        except BrokerError as exc:
            # The stops were just cancelled and the shares are still held. This
            # is the unprotected window, and it stays open until something puts
            # an exit back, so try before reporting.
            restored = self._restore_stop(symbol, cancelled)
            raise BrokerError(
                f"could not close {symbol}: {exc}. "
                + ("The original stop was put back, so the position is "
                   "protected but still open."
                   if restored else
                   f"WARNING: {symbol} is now held with no stop order behind "
                   f"it. Close it by hand or set a stop.")
            ) from exc

        return Fill(symbol, int(float(order.get("qty", 0))), order.get("id", ""),
                    order.get("status", "?"), True)


def committed_symbols(positions, open_orders):
    """Split what the account is already committed to into (filled, working).

    A trading agent must treat both as "already owned". An order that has been
    accepted but has not filled yet still spends buying power and still
    becomes a position at the next open. Deciding what to buy from filled
    positions alone means a second run before the market opens submits the
    same trade again and doubles the risk on it, with no error anywhere.
    """
    filled = {p.get("symbol") for p in (positions or []) if p.get("symbol")}
    working = {o.get("symbol") for o in (open_orders or []) if o.get("symbol")}
    return filled, working


def describe_safety(broker: AlpacaBroker, cfg) -> str:
    """One-line summary of exactly how dangerous the current setup is."""
    if broker.dry_run:
        return "DRY RUN: no orders will be sent anywhere."
    if not broker.is_live:
        return "PAPER: orders go to a fake-money Alpaca account."
    if cfg.allow_live_trading:
        return "LIVE: real money. Orders will be executed."
    return "LIVE endpoint but live trading is disabled in config. Orders will be refused."
