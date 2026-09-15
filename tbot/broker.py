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


# Statuses an order can carry while it is still capable of executing. Kept in
# step with watch.WORKING_STATUSES on purpose rather than imported: the adapter
# must not depend on the watchers, and both are describing one broker fact.
WORKING_STATUSES = ("new", "accepted", "held", "partially_filled", "pending_new",
                    "accepted_for_bidding", "calculated", "")


class AlpacaBroker:
    def __init__(self, key: Optional[str] = None, secret: Optional[str] = None,
                 base_url: Optional[str] = None, dry_run: bool = True):
        self.key = key or os.getenv("ALPACA_API_KEY", "")
        self.secret = secret or os.getenv("ALPACA_API_SECRET", "")
        # `or PAPER_URL` rather than getenv's default, because getenv only
        # applies a default when the variable is ABSENT. GitHub Actions sets an
        # env var to the empty string when the secret behind it does not exist,
        # and "" is not the paper host, so is_live would read True, the
        # live-trading lock would refuse the run, and _cmd_run would return
        # before saving anything: a green build, an unchanged dashboard, and an
        # agent that had silently stopped trading.
        self.base_url = (base_url or os.getenv("ALPACA_BASE_URL")
                         or PAPER_URL).rstrip("/")
        self.dry_run = dry_run

    # -- properties ---------------------------------------------------------

    @property
    def is_live(self) -> bool:
        """Anything not recognisably the paper host counts as live.

        This is the switch all three safety locks hang off, so it has to fail
        in the safe direction. Matching the live URL and calling everything
        else paper means a typo, a stray http://, or a capitalised host is
        silently treated as fake money and the guards never run. Allowing only
        the paper host means the worst case is a refusal, not a real order.
        """
        host = self.base_url.lower()
        for scheme in ("https://", "http://"):
            if host.startswith(scheme):
                host = host[len(scheme):]
                break
        host = host.split("/")[0]
        return host != "paper-api.alpaca.markets"

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
        """Every working order, not the first page of them.

        Alpaca defaults this endpoint to 50 and allows up to 500. With the cap
        at 15 positions, each filled one carrying two live bracket legs and
        each pending entry carrying three orders, 50 is reachable -- and a
        truncated list comes back as an ordinary 200, so it reads as fact. The
        stops that got cut would look missing, protect_exposed would stack a
        second stop behind a position that already had one, and whichever
        filled second would open a naked short. Everything else in this run
        goes out of its way to distinguish "none" from "unknown"; this call
        used to quietly turn one into the other.

        status=all, filtered here, rather than status=open. A bracket's legs
        stay children of the entry order, and once that entry fills the parent
        is no longer open -- so status=open drops the whole group, taking the
        working stop leg with it. The take-profit survives that filter because
        it is live on the exchange in its own right, which is why a fully
        protected bracket read as a naked position with an orphaned target.
        Every alarm about CL, JCI, ABNB, ABT, ADP and AMZN came from here.

        So ask for everything recent and decide what is working locally, using
        the same allowlist the watchers use. nested=true keeps the legs
        attached to their parents and _flatten_orders pulls them back out;
        dedupe is by order id, so a leg arriving both nested and top-level
        counts once.
        """
        out, cutoff = [], None
        seen = set()

        # Working orders first, by their own status, with no date window at
        # all. The paged sweep below walks the 2000 most recently SUBMITTED
        # orders, which is a different set: a GTC protective stop stays working
        # for months while its submitted_at recedes, so on a busy account it
        # eventually falls off the end of that window and the call returns a
        # short list with an ordinary 200. It would read as "this position has
        # no stop", and protect_exposed would stack a second one behind a stop
        # that was working the whole time -- whichever filled first sells the
        # position and the other opens a naked short. status=open cannot drop
        # an order for being old, so it closes that hole; the paged sweep is
        # still needed because status=open omits the legs of a filled parent.
        try:
            live = self._request("GET", "/v2/orders", params={
                "status": "open", "limit": 500, "nested": "true"})
            for o in (live if isinstance(live, list) else []):
                oid = o.get("id")
                if oid and oid in seen:
                    continue
                if oid:
                    seen.add(oid)
                out.append(o)
        except BrokerError as exc:
            # Unknown is not empty, and this list decides whether a position
            # looks protected. The paged sweep below cannot stand in for this
            # read: it walks the 2000 most recently SUBMITTED orders, which is
            # exactly the window an old GTC stop has fallen out of -- the whole
            # reason this request exists. Swallowing a 429 here would hand back
            # a partial order book as though it were complete, the position
            # would read as naked, and the repair would stack a second stop
            # behind a working one. Callers already treat a raise as "stop and
            # say so", which is the safe direction: a missed run costs a day,
            # a doubled stop can open a short.
            raise BrokerError(
                f"could not read the working order list: {exc}") from exc

        for _ in range(4):        # 2000 orders is far more than a day produces
            params = {"status": "all", "limit": 500, "direction": "desc",
                      "nested": "true"}
            if cutoff:
                params["until"] = cutoff
            page = self._request("GET", "/v2/orders", params=params)
            if not isinstance(page, list) or not page:
                break
            for o in page:
                oid = o.get("id")
                if oid and oid in seen:
                    continue
                if oid:
                    seen.add(oid)
                out.append(o)
            if len(page) < 500:
                break
            cutoff = page[-1].get("submitted_at")
            if not cutoff:
                break

        def working(o):
            return str(o.get("status") or "").lower() in WORKING_STATUSES

        # Keep a closed parent whose legs are still working, so the flattening
        # downstream can still reach them. A closed order with nothing live
        # under it is dropped.
        kept = []
        for o in out:
            legs = [l for l in (o.get("legs") or []) if isinstance(l, dict)]
            live_legs = [l for l in legs if working(l)]
            if working(o):
                kept.append(o)
            elif live_legs:
                kept.append(dict(o, legs=live_legs))
        return kept

    def order_history(self, symbols=None, limit: int = 100) -> List[dict]:
        """Every recent order whatever its status, newest first.

        open_orders answers "what is working now", which cannot explain a stop
        that is not working. This answers "what happened to it": an order that
        was created and then cancelled comes back with status cancelled and the
        time it happened, and one that was never created does not come back at
        all. Those need completely different fixes, and nothing else in the
        adapter can tell them apart.
        """
        params = {"status": "all", "limit": min(int(limit), 500),
                  "direction": "desc", "nested": "true"}
        if symbols:
            params["symbols"] = ",".join(sorted(set(symbols)))
        out = self._request("GET", "/v2/orders", params=params)
        return out if isinstance(out, list) else []

    def clock(self) -> dict:
        return self._request("GET", "/v2/clock")

    def activities(self, page_size: int = 100, max_pages: int = 20) -> List[dict]:
        """Raw fill records, newest first, across pages.

        One page holds 100 fills. A position opened more than 100 fills ago has
        no buy inside that window, so entry_dates loses it and the time stop
        silently stops applying to the oldest holdings, which are the only ones
        a time stop was ever going to act on.
        """
        out: List[dict] = []
        token = None
        for _ in range(max_pages):
            params = {"activity_types": "FILL", "page_size": page_size}
            if token:
                params["page_token"] = token
            page = self._request("GET", "/v2/account/activities", params=params)
            if not isinstance(page, list) or not page:
                break
            out.extend(page)
            if len(page) < page_size:
                break
            token = page[-1].get("id")
            if not token:
                break
        return out

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
            side = str(f.get("side") or "")
            # A missing side used to fall through to the sell branch and
            # consume a real lot, inventing a closed trade that never happened.
            # A missing price became 0.00 and printed as a total loss. Neither
            # is worth guessing at: skip the record and stay honest.
            if not side.startswith(("buy", "sell")):
                continue
            if f.get("price") in (None, ""):
                continue
            # This agent is long only. Short-side activity cannot be matched by
            # a long FIFO queue, and trying poisons every trade after it.
            if side in ("sell_short", "buy_to_cover"):
                continue
            try:
                qty = abs(int(float(f.get("qty", 0))))
                price = float(f.get("price"))
            except (TypeError, ValueError):
                continue
            if not sym or qty <= 0 or price <= 0:
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
            # GTC, not day. At Alpaca the time in force applies to the whole
            # bracket, so a "day" bracket takes its stop and target down at the
            # close of the session the entry filled in. The position is then
            # held overnight with no exit working anywhere, which is the exact
            # state this system exists to never be in, and it would happen to
            # every single trade on its first night.
            "time_in_force": "gtc",
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
                    allow_live: bool = False, acknowledged: bool = False,
                    last_price: Optional[float] = None) -> Fill:
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
        if last_price is not None and stop >= last_price:
            # A sell stop at or above the market is not protection, it is an
            # instruction to dump the position immediately.
            return Fill(symbol, 0, "", "rejected", False,
                        f"stop {stop:.2f} is not below the last price "
                        f"{last_price:.2f}")

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

    def cancel_order_ids(self, order_ids) -> List[str]:
        """Cancel specific orders and return the ids that are now gone.

        cancel_orders_for takes a whole symbol, which is right before a close
        and wrong when one order is in the way and the others are protection.
        An order that has already filled or been cancelled counts as gone: it
        is not working any more, which is the only thing the caller asked.
        """
        if self.dry_run:
            return []
        gone: List[str] = []
        for oid in order_ids:
            if not oid:
                continue
            try:
                self._request("DELETE", f"/v2/orders/{oid}")
            except BrokerError:
                pass   # filled or cancelled between reading and deleting
            gone.append(str(oid))
        return gone

    def submit_protective_oco(self, symbol: str, shares: int, stop: float,
                              target: float, allow_live: bool = False,
                              acknowledged: bool = False,
                              last_price: Optional[float] = None) -> Fill:
        """Put a linked stop and take-profit behind a position that has neither.

        One-cancels-other, so the pair behaves like the bracket legs it is
        replacing: whichever fills removes the other. Submitting the two as
        separate orders instead would leave the survivor working after the
        position is closed, and a resting sell against no shares is how a long
        becomes a short.
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
        if stop <= 0 or target <= 0:
            return Fill(symbol, 0, "", "rejected", False, "stop or target is not positive")
        if target <= stop:
            return Fill(symbol, 0, "", "rejected", False,
                        f"target {target:.2f} is not above stop {stop:.2f}")
        if last_price is not None:
            # Either leg on the wrong side of the market fills the moment it is
            # accepted, which closes the position instead of protecting it.
            if stop >= last_price:
                return Fill(symbol, 0, "", "rejected", False,
                            f"stop {stop:.2f} is not below the last price {last_price:.2f}")
            if target <= last_price:
                return Fill(symbol, 0, "", "rejected", False,
                            f"target {target:.2f} is not above the last price {last_price:.2f}")

        if self.dry_run:
            return Fill(symbol, shares, "", "dry-run", False,
                        f"would protect {shares} {symbol} with a stop at "
                        f"{stop:.2f} and a target at {target:.2f}")

        order = self._request("POST", "/v2/orders", json={
            "symbol": symbol,
            "qty": str(shares),
            "side": "sell",
            "type": "limit",
            "limit_price": round(target, 2),
            "time_in_force": "gtc",
            "order_class": "oco",
            "take_profit": {"limit_price": round(target, 2)},
            "stop_loss": {"stop_price": round(stop, 2)},
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

        # Flatten first. open_orders() returns bracket legs NESTED under their
        # parent, and after the entry fills the parent is the only thing at the
        # top level -- a filled order, which Alpaca will not cancel. Iterating
        # the top level therefore attempted exactly one DELETE, got a 422 for
        # trying to cancel a fill, swallowed it, and returned an empty list
        # while both live legs kept reserving the shares. The close that
        # followed was then rejected for insufficient quantity, _restore_stop
        # found no sell order to read a price from, and the run reported a
        # fully protected position as naked. Every soft exit -- trend break,
        # time stop, reversion exit -- failed this way on any position opened
        # by a bracket, which is all of them.
        killed: List[dict] = []
        from tbot.watch import WORKING_STATUSES, _flatten_orders
        for o in _flatten_orders(self.open_orders()):
            if o.get("symbol") != symbol or not o.get("id"):
                continue
            # Only working orders can be cancelled. Asking Alpaca to cancel a
            # fill is a guaranteed 422 that tells us nothing.
            if str(o.get("status") or "").lower() not in WORKING_STATUSES:
                continue
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
        except BrokerError as exc:
            # Unknown is not "no positions have an entry date". Every time stop
            # silently stops applying when this call fails, and the caller had
            # no way to tell that from "these positions are all new". The
            # reversion strategy feels it worst: its only exits are a close
            # above the 10-day average and a 10-session time stop, so a trade
            # that never bounces has no soft exit left at all.
            raise BrokerError(f"could not read fill history: {exc}") from exc
        open_qty: Dict[str, int] = {}
        for f in fills:
            sym, side = f.get("symbol"), f.get("side", "")
            when = (f.get("transaction_time") or "")[:10]
            if not sym or not when:
                continue
            try:
                qty = abs(int(float(f.get("qty") or 0)))
            except (TypeError, ValueError):
                qty = 0
            if side.startswith("buy"):
                if open_qty.get(sym, 0) <= 0:
                    out[sym] = when          # a fresh position starts the clock
                open_qty[sym] = open_qty.get(sym, 0) + qty
            elif side.startswith("sell"):
                # Only a sale that closes the whole position resets the clock.
                # Scaling out of half a position used to wipe the entry date,
                # which quietly exempted the remaining shares from the time
                # stop for as long as they were held.
                open_qty[sym] = open_qty.get(sym, 0) - qty
                if open_qty.get(sym, 0) <= 0:
                    open_qty[sym] = 0
                    out.pop(sym, None)
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
