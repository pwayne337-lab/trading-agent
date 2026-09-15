"""
The scoreboard.

Builds a single self-contained HTML page from the agent's recorded state.
No server, no database, no JavaScript framework. Everything the page shows
came from a real run, so the page cannot report a trade that did not happen.

The most important thing on it is the freshness banner. A dashboard that
quietly shows three-day-old numbers while you believe they are live is worse
than no dashboard at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .report import _line_chart
from .state import load_equity_history, load_state

SITE = Path(__file__).resolve().parent.parent / "site"
SITE.mkdir(parents=True, exist_ok=True)


def _age(updated_at) -> tuple:
    """Returns (human string, hours, severity)."""
    if not updated_at:
        return "never run", 1e9, "critical"
    try:
        t = datetime.fromisoformat(str(updated_at))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
    except Exception:
        return "unknown", 1e9, "critical"

    hrs = (datetime.now(timezone.utc) - t).total_seconds() / 3600
    # A run stamped slightly ahead of this clock is clock skew, not a run from
    # the future. Never render it as negative minutes.
    if hrs < 0:
        hrs = 0.0
    if hrs < 1:
        s = f"{int(hrs * 60)} min ago"
    elif hrs < 48:
        n = int(hrs)
        s = f"{n} {'hour' if n == 1 else 'hours'} ago"
    else:
        s = f"{int(hrs / 24)} days ago"

    sev = "good" if hrs < 30 else ("warning" if hrs < 96 else "critical")
    return s, hrs, sev


def _tile(label, value, sub="", tone="", ids=""):
    """`ids` is set on the one tile JavaScript has to keep updating: the age
    of the last run changes every minute the page sits open, and a value
    baked in at build time would be wrong the moment after it was written."""
    attr = f' id="{ids}"' if ids else ""
    val_id = ' id="age-val"' if ids == "age-tile" else ""
    return (f'<div class="tile {tone}"{attr}><div class="tl">{label}</div>'
            f'<div class="tv"{val_id}>{value}</div><div class="ts">{sub}</div></div>')


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_html(state: dict = None, history: list = None,
               title: str = "Trading agent") -> str:
    state = state if state is not None else load_state()
    history = history if history is not None else load_equity_history()

    acct = state.get("account") or {}
    equity = float(acct.get("equity") or 0)
    cash = float(acct.get("cash") or 0)
    mode = str(state.get("mode", "unknown")).upper()
    positions = state.get("positions") or []
    signals = state.get("signals") or []
    vetoes = state.get("vetoes") or []
    orders = state.get("orders") or []
    skipped = state.get("skipped") or []
    errors = state.get("errors") or []

    # When a run stopped early the figures below were carried over from an
    # earlier run, so they must be dated by that run and not by the aborted one
    # that happened to rebuild the page. Otherwise a page full of week-old
    # numbers reports itself as minutes old, which is the one thing a
    # monitoring page must never do.
    carried_from = state.get("carried_from")
    age_str, age_hrs, age_sev = _age(carried_from or state.get("updated_at"))

    # Day change from the equity history.
    day_change = 0.0
    if len(history) >= 2 and history[-2]["equity"] > 0:
        day_change = (history[-1]["equity"] / history[-2]["equity"] - 1) * 100
    total_change = 0.0
    if history and history[0]["equity"] > 0:
        total_change = (equity / history[0]["equity"] - 1) * 100

    open_pl = sum(float(p.get("unrealized_pl") or 0) for p in positions)

    # -- banner -------------------------------------------------------------
    if not state.get("updated_at"):
        banner = ('<div class="banner warning"><strong>Nothing here yet.</strong> '
                  'The agent has not run. This page fills in the first time it does.</div>')
    elif age_sev == "critical":
        banner = (f'<div class="banner critical"><strong>Stale.</strong> '
                  f'The agent last ran {age_str}. Nothing on this page is current. '
                  f'Check that the scheduled run is still working.</div>')
    elif age_sev == "warning":
        banner = (f'<div class="banner warning"><strong>Getting old.</strong> '
                  f'Last run {age_str}. Fine over a weekend or a market holiday, '
                  f'not fine on a Wednesday.</div>')
    elif state.get("monitor_only"):
        # A mid-session check refreshed the account and the watchers, but it did
        # not scan and it placed no trades. The orders and signals below are
        # still the trading run's, and without saying so this page would show
        # them under a lunchtime timestamp as though they had just happened.
        banner = ('<div class="banner info"><strong>Mid-session check.</strong> '
                  'Positions, orders and the safety checks below were refreshed '
                  'just now. Any signals and orders shown are from the last '
                  'after-close run; this check does not trade.</div>')
    elif state.get("refresh_only"):
        banner = ('<div class="banner info"><strong>Live figures.</strong> '
                  'The account below was re-read just now. Signals, orders and '
                  'checks are from the last trading run; this refresh does not '
                  'run the strategy.</div>')
    else:
        banner = ""

    # How long since the trading logic actually ran, which is a different
    # question from how old the figures are. The refresh job re-reads the
    # account every hour, so the staleness banner above is green almost
    # always -- and on its own it would have said everything was fine right
    # through the three days in September when the agent was not running at
    # all. This is the banner that would have caught it.
    _last_full = state.get("last_full_run")
    if state.get("updated_at"):
        _full_str, _full_hrs, _ = _age(_last_full)
        if not _last_full:
            banner += ('<div class="banner critical"><strong>The agent has not '
                       'traded.</strong> These figures are current, but no '
                       'trading run is on record. The numbers are real; the '
                       'agent behind them is not running.</div>')
        elif _full_hrs >= 72:
            banner += (f'<div class="banner critical"><strong>Not trading.</strong> '
                       f'The figures below are current, but the agent last ran '
                       f'{_esc(_full_str)}. Something is stopping the scheduled '
                       f'run: check the Actions tab.</div>')
        elif _full_hrs >= 30:
            banner += (f'<div class="banner warning"><strong>No run since '
                       f'{_esc(_full_str)}.</strong> The account below is current, '
                       f'but the trading logic has not run. Fine over a weekend '
                       f'or a holiday, not fine on a Wednesday.</div>')

    if carried_from:
        why = _esc(state.get("carried_reason") or "the run stopped early")
        banner += (f'<div class="banner warning"><strong>Carried forward.</strong> '
                   f'The last run stopped before it read the account, because '
                   f'{why}. Every figure below is from the run of '
                   f'{_esc(carried_from)} UTC and has not been re-measured '
                   f'since. Nothing about the account has changed; only this '
                   f'page is waiting on a clean run.</div>')

    # The watchers come first. A missing stop order matters more than the
    # equity number sitting under it.
    findings = state.get("findings") or []
    # A finding is a judgement about the broker at a moment in time, and when
    # the figures were carried forward these came with them. Saying "needs
    # attention" with no date next to numbers that are openly stale reads as a
    # live alarm, and a red banner nobody can date is one nobody can act on.
    when = f" as of {_esc(str(carried_from)[:16])} UTC" if carried_from else ""
    for f in findings:
        sev = f.get("severity")
        if sev not in ("critical", "warning"):
            continue
        cls = "critical" if sev == "critical" else "warning"
        label = ("Needed attention" + when + "." if sev == "critical"
                 else "Worth a look" + when + ".")
        if not carried_from:
            label = "Needs attention." if sev == "critical" else "Worth a look."
        banner += (f'<div class="banner {cls}"><strong>{label}</strong> '
                   f'{_esc(f.get("message"))}</div>')

    # Errors are shown whenever there are errors. This used to be gated on
    # `not findings`, which was meant to avoid saying the same thing twice --
    # but the loop above renders only critical and warning findings, so a
    # findings list holding nothing but a routine INFO note suppressed the
    # error banner entirely. `check_broker` emits exactly such a note ("new
    # since last run: ...") on any day the position set changed, which is most
    # days something happens. The result was that "could not place a stop on
    # NVDA" rendered as a completely clean page.
    if errors:
        banner += (f'<div class="banner critical"><strong>Errors on the '
                   f'{"carried-forward" if carried_from else "last"} run.</strong> '
                   + "; ".join(_esc(e) for e in errors[:3]) + '</div>')

    if mode == "LIVE":
        banner = ('<div class="banner critical"><strong>LIVE MONEY.</strong> '
                  'This agent is trading a funded account.</div>') + banner

    # -- charts -------------------------------------------------------------
    if len(history) >= 2:
        dates = [h["date"][5:] for h in history]
        vals = [h["equity"] for h in history]
        chart, pts = _line_chart(dates, vals, "--series-1", fill=True)
        chart_block = f'<div class="card" id="c1">{chart}<div class="tip" id="t1"></div></div>'
    else:
        chart, pts = "", "[]"
        chart_block = ('<div class="card empty">Not enough history to plot yet. '
                       'The chart appears after the agent has run on two separate days.</div>')

    # -- tiles --------------------------------------------------------------
    tiles = "".join([
        _tile("Account equity", f"${equity:,.2f}",
              f"{mode.lower()} account &middot; {len(history)} run(s) recorded"),
        _tile("Today", f"{day_change:+.2f}%", "since the previous run",
              "good" if day_change > 0 else ("bad" if day_change < 0 else "")),
        _tile("Since start", f"{total_change:+.2f}%", "of tracking",
              "good" if total_change > 0 else ("bad" if total_change < 0 else "")),
        _tile("Open positions", f"{len(positions)}",
              f"open P&amp;L ${open_pl:+,.2f}",
              "good" if open_pl > 0 else ("bad" if open_pl < 0 else "")),
        _tile("Cash", f"${cash:,.2f}", "uninvested"),
        _tile("Last run", age_str,
              f"{len(orders)} order(s), {len(vetoes)} blocked", age_sev,
              ids="age-tile"),
    ])

    # -- positions ----------------------------------------------------------
    if positions:
        rows = "".join(
            f"<tr><td><strong>{_esc(p['symbol'])}</strong></td>"
            f"<td class='n'>{p.get('shares', 0)}</td>"
            f"<td class='n'>${float(p.get('avg_entry') or 0):,.2f}</td>"
            f"<td class='n'>${float(p.get('market_value') or 0):,.2f}</td>"
            f"<td class='n {'pos' if float(p.get('unrealized_pl') or 0) >= 0 else 'neg'}'>"
            f"${float(p.get('unrealized_pl') or 0):+,.2f}</td></tr>"
            for p in positions)
        pos_block = (f'<div class="card scroll positions-summary"><table><thead><tr><th>Symbol</th>'
                     f'<th class="n">Shares</th><th class="n">Avg cost</th>'
                     f'<th class="n">Value</th><th class="n">Open P&amp;L</th>'
                     f'</tr></thead><tbody>{rows}</tbody></table></div>')
    else:
        pos_block = '<div class="card empty">Holding nothing right now.</div>'

    # -- today --------------------------------------------------------------
    items = []
    # Repairs go first. When the agent both reports a problem and fixes it,
    # the page has to say so, or the finding below reads as an open emergency
    # that nobody dealt with.
    for r in (state.get("protected") or []):
        items.append(
            f'<li class="ev fixed"><span class="tag">protected</span> '
            f'<strong>{_esc(r.get("symbol"))}</strong> had no stop behind it. '
            f'Placed one on {r.get("shares", 0)} sh at '
            f'${float(r.get("stop") or 0):,.2f}.</li>')
    for e in (state.get("exits") or []):
        pl = e.get("unrealized_pl")
        pl_txt = f", P&amp;L ${float(pl):+,.2f}" if pl is not None else ""
        items.append(
            f'<li class="ev sold"><span class="tag">sold</span> '
            f'<strong>{_esc(e.get("symbol"))}</strong> {_esc(e.get("reason"))}'
            f'{pl_txt}</li>')
    for o in orders:
        # Which rule set opened it. Without this the page shows a list of buys
        # with no way to tell a breakout from a dip buy, and no way to notice
        # that one strategy has been carrying everything.
        via = o.get("strategy")
        via_txt = f' <span class="tag">{_esc(via)}</span>' if via else ""
        items.append(
            f'<li class="ev ok"><span class="tag">bought</span> '
            f'<strong>{_esc(o.get("symbol"))}</strong>{via_txt} '
            f'{o.get("shares", 0)} sh, '
            f'stop ${float(o.get("stop") or 0):,.2f}, '
            f'target ${float(o.get("target") or 0):,.2f}, '
            f'risking ${float(o.get("dollars_at_risk") or 0):,.2f}</li>')
    for v in vetoes:
        items.append(
            f'<li class="ev block"><span class="tag">blocked</span> '
            f'<strong>{_esc(v.get("symbol"))}</strong> {_esc(v.get("reason"))}</li>')
    for s in skipped:
        items.append(
            f'<li class="ev skip"><span class="tag">skipped</span> '
            f'<strong>{_esc(s.get("symbol"))}</strong> {_esc(s.get("reason"))}</li>')
    if not items:
        items.append('<li class="ev quiet">No setups met the rules. '
                     'Most days look like this.</li>')
    today_block = f'<div class="card"><ul class="events">{"".join(items)}</ul></div>'

    # -- briefing -----------------------------------------------------------
    brief = state.get("briefing") or ""
    if brief:
        paras = "".join(f"<p>{_esc(p)}</p>" for p in brief.split("\n") if p.strip())
        brief_block = f'<div class="card brief">{paras}</div>'
    else:
        brief_block = ('<div class="card empty">No briefing. Add an Anthropic API '
                       'key to .env to turn this on.</div>')

    # -- recent trades ------------------------------------------------------
    rt = state.get("recent_trades") or []
    if rt:
        rows = "".join(
            f"<tr><td><strong>{_esc(t.get('symbol'))}</strong></td>"
            f"<td>{_esc(t.get('closed'))}</td>"
            f"<td>{_esc(t.get('reason'))}</td>"
            f"<td class='n {'pos' if float(t.get('pnl') or 0) >= 0 else 'neg'}'>"
            f"${float(t.get('pnl') or 0):+,.2f}</td></tr>" for t in rt[:20])
        trades_block = (f'<div class="card scroll"><table><thead><tr><th>Symbol</th>'
                        f'<th>Closed</th><th>Why</th><th class="n">P&amp;L</th>'
                        f'</tr></thead><tbody>{rows}</tbody></table></div>')
    else:
        trades_block = '<div class="card empty">No closed trades yet.</div>'

    # -- positions in detail (its own tab) -----------------------------------
    # The overview answers "what do I hold and is it up". This answers the
    # question you actually act on: how much of that is still at risk, and how
    # far the price has to fall before the agent is out.
    strat_of = state.get("strategy_by_symbol") or {}
    if positions:
        rows = []
        for p in positions:
            sym = _esc(p.get("symbol"))
            shares = float(p.get("shares") or 0)
            entry = float(p.get("avg_entry") or 0)
            value = float(p.get("market_value") or 0)
            pl = float(p.get("unrealized_pl") or 0)
            last = value / shares if shares else 0.0
            stop = p.get("stop")
            target = p.get("target")
            strat = _esc(strat_of.get(p.get("symbol"), ""))

            if stop:
                stop = float(stop)
                to_stop = (last - stop) / last * 100 if last else 0.0
                at_risk = max(0.0, (last - stop) * shares)
                initial = (entry - stop) * shares
                r_mult = (pl / initial) if initial > 0 else None
                stop_cell = f"${stop:,.2f}"
                dist_cell = f"{to_stop:,.1f}%"
                risk_cell = f"${at_risk:,.0f}"
                r_cell = f"{r_mult:+.2f}R" if r_mult is not None else "&mdash;"
                cov = int(p.get("stop_shares") or 0)
                if cov and shares and cov < shares:
                    stop_cell += (f" <span class='warn-inline'>covers {cov} of "
                                  f"{int(shares)}</span>")
            elif p.get("stop_checked"):
                # Not cosmetic. No working stop is the one thing on this page
                # that needs acting on today, so it is said in words.
                stop_cell = "<span class='warn-inline'>none</span>"
                dist_cell = risk_cell = r_cell = "&mdash;"
            else:
                # The order book was never read for this position, so nothing
                # is known either way. Saying "none" here would be an alarm
                # invented out of a missing field.
                stop_cell = "<span class='muted-inline'>not checked</span>"
                dist_cell = risk_cell = r_cell = "&mdash;"

            tag = f' <span class="tag">{strat}</span>' if strat else ""
            # `sec` marks the columns a phone drops. Nine columns do not fit on
            # a 390px screen, and the ones that were scrolling off were Stop,
            # At risk and P&L -- which is the entire question you open this on
            # a phone to answer. Shares, entry and last price are reference,
            # and they stay one turn of the device away rather than pushing
            # the answer off the edge.
            rows.append(
                f"<tr><td><strong>{sym}</strong>{tag}</td>"
                f"<td class='n sec'>{int(shares)}</td>"
                f"<td class='n sec'>${entry:,.2f}</td>"
                f"<td class='n sec'>${last:,.2f}</td>"
                f"<td class='n'>{stop_cell}</td>"
                f"<td class='n'>{dist_cell}</td>"
                f"<td class='n sec'>{risk_cell}</td>"
                f"<td class='n sec'>{r_cell}</td>"
                f"<td class='n {'pos' if pl >= 0 else 'neg'}'>${pl:+,.2f}</td></tr>")
        detail_block = (
            '<div class="card scroll positions-detail"><table><thead><tr><th>Symbol</th>'
            '<th class="n sec">Shares</th><th class="n sec">Entry</th>'
            '<th class="n sec">Last</th>'
            '<th class="n">Stop</th><th class="n">To stop</th>'
            '<th class="n sec">At risk</th><th class="n sec">R</th>'
            '<th class="n">Open P&amp;L</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div>'
            + '<p class="note">"At risk" is what this position loses from here if '
              'its stop is hit. "R" is the profit measured in units of the risk '
              'originally taken, so +1R means it has made back exactly what it '
              'was prepared to lose.</p>')
    else:
        detail_block = '<div class="card empty">Holding nothing right now.</div>'

    # -- performance (its own tab) -------------------------------------------
    # Every number here is counted from closed trades only. Open positions have
    # no result yet, and mixing them in is how a losing run gets told as a
    # winning one.
    closed = [t for t in (state.get("recent_trades") or [])
              if t.get("pnl") is not None]
    if closed:
        pnls = [float(t.get("pnl") or 0) for t in closed]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x < 0]
        gross_win, gross_loss = sum(wins), abs(sum(losses))
        pf = (gross_win / gross_loss) if gross_loss else None
        stat_tiles = "".join([
            _tile("Closed trades", str(len(closed))),
            _tile("Win rate", f"{len(wins) / len(closed) * 100:,.0f}%",
                  f"{len(wins)} of {len(closed)}"),
            _tile("Net P&L", f"${sum(pnls):+,.2f}",
                  tone="good" if sum(pnls) >= 0 else "bad"),
            _tile("Average win", f"${(gross_win / len(wins)) if wins else 0:,.2f}"),
            _tile("Average loss", f"${-(gross_loss / len(losses)) if losses else 0:,.2f}"),
            _tile("Profit factor", f"{pf:,.2f}" if pf else "&mdash;",
                  "made per $1 lost"),
        ])
        by = {}
        for t in closed:
            key = str(t.get("reason") or "unknown")
            b = by.setdefault(key, {"n": 0, "pnl": 0.0})
            b["n"] += 1
            b["pnl"] += float(t.get("pnl") or 0)
        by_rows = "".join(
            f"<tr><td>{_esc(k)}</td><td class='n'>{v['n']}</td>"
            f"<td class='n {'pos' if v['pnl'] >= 0 else 'neg'}'>"
            f"${v['pnl']:+,.2f}</td></tr>"
            for k, v in sorted(by.items(), key=lambda kv: -abs(kv[1]["pnl"])))
        perf_block = (
            f'<div class="tiles">{stat_tiles}</div>'
            f'<h3>By exit reason</h3>'
            f'<div class="card scroll"><table><thead><tr><th>Why it closed</th>'
            f'<th class="n">Trades</th><th class="n">P&amp;L</th></tr></thead>'
            f'<tbody>{by_rows}</tbody></table></div>'
            f'<p class="note">Counted from the {len(closed)} closed trades on '
            f'record. That is far too few to judge a strategy by &mdash; a run '
            f'of luck either way swamps it at this size. Treat it as a record '
            f'of what happened, not evidence of what will.</p>')
    else:
        perf_block = ('<div class="card empty">No closed trades yet. This fills '
                      'in as positions are exited.</div>')

    # -- the rules (its own tab) ---------------------------------------------
    # Generated from the live config object, not written out by hand, so it
    # cannot drift from what the agent actually does. Change a setting and this
    # page changes with it.
    try:
        from tbot.config import AgentConfig
        from tbot.strategy import PLAYBOOK
        cfg = AgentConfig()
        r, sc = cfg.risk, cfg.strategy

        def _rule(label, value, why):
            return (f'<tr><td>{_esc(label)}</td><td class="n"><strong>'
                    f'{_esc(value)}</strong></td><td class="why">{_esc(why)}</td></tr>')

        money = "".join([
            _rule("Risked per trade", f"{r.risk_per_trade * 100:g}% of the account",
                  "A full stop-out costs this much of the account, not this much "
                  "of the position. Every position is sized backwards from it."),
            _rule("Largest single position", f"{r.max_position_pct * 100:g}% of equity",
                  "A very tight stop would otherwise size you into a position so "
                  "large that one overnight gap does real damage."),
            _rule("Most positions at once", str(r.max_open_positions),
                  "A ceiling on how many things can go wrong at the same time."),
            _rule("Total invested", f"up to {r.max_gross_exposure * 100:g}% of equity",
                  "Never borrows. At 100% the account can be fully invested but "
                  "not leveraged."),
            _rule("Stops trading if down", f"{r.max_drawdown_halt * 100:g}%",
                  f"Opens nothing new until the account recovers to within "
                  f"{r.resume_below * 100:g}% of its high, and waits "
                  f"{r.halt_cooldown_days} days before reconsidering."),
            _rule("Refuses lookalikes", f"{r.max_correlation:g} correlation",
                  f"Two things that move together are one bet in two names. "
                  f"Measured on {r.correlation_window} days of returns."),
        ])

        plays = "".join(
            f'<tr><td><strong>{_esc(name)}</strong></td>'
            f'<td class="why">{_esc(getattr(spec, "summary", "") or "")}</td></tr>'
            for name, spec in PLAYBOOK.items()
            if name in (sc.enabled or []))

        mech = "".join([
            _rule("Trend filter", f"{sc.sma_fast}-day over {sc.sma_slow}-day average",
                  "Only buys things already trending up on the slower measure."),
            _rule("Stop distance", f"{sc.min_stop_atr:g}-{sc.max_stop_atr:g} ATR",
                  f"Placed off recent volatility ({sc.atr_period}-day ATR) rather "
                  f"than a fixed percentage, so a calm stock gets a tight stop "
                  f"and a wild one gets room."),
            _rule("Reward sought", f"{sc.reward_risk:g}x the risk",
                  "The take-profit sits this many multiples of the stop distance above entry."),
            _rule("Maximum hold", f"{sc.max_hold_days} sessions",
                  "A trade that has gone nowhere for this long is closed and the "
                  "slot given back."),
            _rule("Minimum liquidity",
                  f"${sc.min_avg_dollar_volume:,.0f} traded a day",
                  "Below this the spread and the slippage eat the edge."),
        ])

        costs = _rule("Assumed costs",
                      f"{cfg.costs.slippage_bps:g} bps entry, "
                      f"{cfg.costs.stop_slippage_bps:g} bps on stops",
                      "Charged in every backtest, so the tested result is after "
                      "costs rather than before them.")

        rules_block = (
            '<p class="note">Everything below is read from the agent\'s live '
            'settings when this page is built, so it always describes what the '
            'agent is actually doing right now.</p>'
            '<h3>Money and risk</h3>'
            '<div class="card scroll"><table><thead><tr><th>Rule</th>'
            '<th class="n">Setting</th><th>Why</th></tr></thead>'
            f'<tbody>{money}</tbody></table></div>'
            '<h3>What it looks for</h3>'
            '<div class="card scroll"><table><thead><tr><th>Strategy</th>'
            f'<th>What it buys</th></tr></thead><tbody>{plays}</tbody></table></div>'
            '<h3>How it decides</h3>'
            '<div class="card scroll"><table><thead><tr><th>Rule</th>'
            '<th class="n">Setting</th><th>Why</th></tr></thead>'
            f'<tbody>{mech}{costs}</tbody></table></div>')
    except Exception as exc:     # a broken settings page must not break the page
        rules_block = (f'<div class="card empty">Could not read the settings '
                       f'({_esc(type(exc).__name__)}).</div>')

    # -- one verdict, above everything ---------------------------------------
    # The page could already tell you everything was wrong; it could not tell
    # you that nothing was. Checking on an agent should take one glance, and
    # "no news" has to be stated rather than inferred from the absence of red.
    crit = [f for f in findings if f.get("severity") == "critical"]
    warn = [f for f in findings if f.get("severity") == "warning"]
    _lf_str, _lf_hrs, _ = _age(state.get("last_full_run"))
    troubles = []
    if errors:
        troubles.append(f"{len(errors)} error{'s' if len(errors) > 1 else ''} "
                        f"on the last run")
    if crit:
        troubles.append(f"{len(crit)} critical check{'s' if len(crit) > 1 else ''}")
    if state.get("last_full_run") and _lf_hrs >= 30:
        troubles.append(f"no trading run since {_lf_str}")
    if not state.get("last_full_run"):
        troubles.append("no trading run on record")
    # Only positions whose order book was actually read. A position with no
    # stop recorded because the page predates the field is not a position
    # without a stop, and this alarm is far too important to fire on a guess.
    naked = [p for p in positions
             if p.get("stop_checked") and not p.get("stop")]
    if naked:
        troubles.append(f"{len(naked)} position{'s' if len(naked) > 1 else ''} "
                        f"with no stop")

    if troubles:
        # An agent that has not traded in three days is as serious as an error,
        # and the banner below already calls it critical. Two parts of one page
        # disagreeing about how bad something is teaches you to trust neither.
        _dead = (not state.get("last_full_run")) or _lf_hrs >= 72
        verdict_tone = "critical" if (errors or crit or naked or _dead) else "warning"
        verdict_head = "Needs you"
        verdict_text = "; ".join(troubles) + "."
        verdict_sub = ("Open the report below and send it to Claude."
                       if verdict_tone == "critical" else
                       "Worth a look when you get a chance.")
    elif warn:
        verdict_tone = "warning"
        verdict_head = "Running, with a note"
        verdict_text = f"{len(warn)} thing{'s' if len(warn) > 1 else ''} worth a look."
        verdict_sub = "Nothing is broken."
    else:
        verdict_tone = "ok"
        verdict_head = "All good"
        verdict_text = "The agent is running and nothing needs you."
        verdict_sub = (f"Last traded {_lf_str}."
                       if state.get("last_full_run") else "")

    verdict_block = (f'<div class="verdict {verdict_tone}">'
                     f'<span class="dot"></span>'
                     f'<div><strong>{_esc(verdict_head)}</strong> '
                     f'{_esc(verdict_text)}'
                     + (f'<span class="vsub">{_esc(verdict_sub)}</span>'
                        if verdict_sub else "")
                     + '</div></div>')

    # -- a report worth pasting ----------------------------------------------
    # The loop this is built for: something goes wrong, the page says so, and
    # the whole picture gets handed to Claude in one paste. Assembled here
    # rather than left to be described from memory, because the details that
    # matter are the ones nobody thinks to mention.
    lines = ["AGENT DIAGNOSTIC REPORT",
             f"page built     : {state.get('updated_at') or 'never'}",
             f"figures from   : {carried_from or state.get('updated_at') or 'never'}",
             f"last full run  : {state.get('last_full_run') or 'NONE ON RECORD'}",
             f"mode           : {mode}",
             f"healthy flag   : {state.get('healthy')}",
             f"this write was : " + ("an hourly refresh" if state.get("refresh_only")
                                     else "a mid-session check" if state.get("monitor_only")
                                     else "a trading run"),
             f"equity         : {acct.get('equity')}",
             f"cash           : {acct.get('cash')}",
             f"positions      : {len(positions)}"]
    if carried_from:
        lines.append(f"carried because: {state.get('carried_reason')}")
    lines.append("")
    lines.append(f"ERRORS ({len(errors)}):")
    lines += [f"  - {e}" for e in errors] or ["  none"]
    lines.append("")
    lines.append(f"CHECKS ({len(findings)}):")
    lines += [f"  [{f.get('severity')}] {f.get('agent')}: {f.get('message')}"
              for f in findings] or ["  none"]
    lines.append("")
    lines.append("POSITIONS:")
    if positions:
        for p in positions:
            stop = p.get("stop")
            lines.append(
                f"  {p.get('symbol')}: {p.get('shares')} sh @ "
                f"{p.get('avg_entry')}, stop "
                f"{stop if stop else 'NONE'}, "
                f"P/L {p.get('unrealized_pl')}, "
                f"opened by {(state.get('strategy_by_symbol') or {}).get(p.get('symbol'), '?')}")
    else:
        lines.append("  none")
    lines.append("")
    lines.append("SCHEDULE (UTC, weekdays):")
    lines += ["  14-21 hourly  refresh the page (no trading)",
              "  15:00         mid-session check + protect",
              "  16:00         protect again",
              "  21:30         the trading run"]
    lines.append("")
    lines.append("Paste this to Claude with what you saw on the page.")
    report_text = "\n".join(lines)
    report_esc = _esc(report_text)

    stamp = (f"Generated {state['updated_at']} UTC from the agent's own run records"
             if state.get("updated_at") else "This page has not been generated from a real run yet")
    if carried_from:
        stamp += f", carrying the figures from the run of {carried_from} UTC"
    # Two timestamps, because they answer two different questions. DATED is how
    # old the numbers on the page are, and drives the age tile and the stale
    # banner. BUILT is which run wrote this file, and is compared against
    # data.json to notice a newer run. Collapsing them would either misdate the
    # figures or send the browser to reload a page it is already showing.
    built_iso = str(state.get("updated_at") or "")
    dated_iso = str(carried_from or state.get("updated_at") or "")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<!-- Added to a phone home screen this runs as a standalone app, and iOS will
     happily serve the copy it cached the day you added it. The script at the
     bottom checks for a newer run and reloads past that cache. -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Agent">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#070b12">
<link rel="manifest" href="./manifest.webmanifest">
<link rel="icon" href="./app-icon.svg" type="image/svg+xml">
<meta http-equiv="Cache-Control" content="no-cache, must-revalidate">
<title>{_esc(title)}</title>
<style>
  :root {{
    color-scheme:light;
    --plane:#f3f5f8; --surface-1:#ffffff; --surface-2:#f7f9fc;
    --surface-raised:#ffffff; --text-primary:#0b1220; --text-secondary:#48556a;
    --muted:#7b8799; --gridline:#e8edf3; --border:rgba(15,23,42,.09);
    --series-1:#2563eb; --series-soft:rgba(37,99,235,.10);
    --good:#078352; --good-soft:rgba(7,131,82,.10);
    --critical:#d23f4c; --critical-soft:rgba(210,63,76,.10);
    --warning:#aa6800; --warning-soft:rgba(170,104,0,.10);
    --shadow:0 1px 2px rgba(15,23,42,.04),0 12px 32px rgba(15,23,42,.06);
    --nav-height:68px;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --plane:#070b12; --surface-1:#0d131d; --surface-2:#111a27;
      --surface-raised:#151e2c; --text-primary:#f6f8fb; --text-secondary:#aab5c5;
      --muted:#738096; --gridline:#1b2635; --border:rgba(255,255,255,.085);
      --series-1:#62a1ff; --series-soft:rgba(98,161,255,.11);
      --good:#35d49a; --good-soft:rgba(53,212,154,.10);
      --critical:#ff6b75; --critical-soft:rgba(255,107,117,.10);
      --warning:#f3b54a; --warning-soft:rgba(243,181,74,.10);
      --shadow:0 1px 2px rgba(0,0,0,.25),0 18px 45px rgba(0,0,0,.22);
    }}
  }}
  *{{box-sizing:border-box}}
  html{{scroll-behavior:smooth}}
  body{{margin:0;min-height:100vh;background:var(--plane);color:var(--text-primary);
    font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
    -webkit-text-size-adjust:100%;font-variant-numeric:tabular-nums}}
  body::before{{content:"";position:fixed;inset:0 0 auto;height:360px;pointer-events:none;
    background:radial-gradient(700px 300px at 50% -100px,var(--series-soft),transparent 72%);
    z-index:-1}}
  button,a{{-webkit-tap-highlight-color:transparent}}
  .wrap{{max-width:1120px;margin:0 auto;padding:30px 28px 72px}}
  header{{display:flex;gap:16px;align-items:center;justify-content:space-between;
    margin-bottom:24px}}
  .brand{{display:flex;align-items:center;gap:12px;min-width:0}}
  .brandmark{{width:38px;height:38px;display:grid;place-items:center;flex:0 0 auto;
    border-radius:12px;color:white;background:linear-gradient(145deg,#397cff,#2352ce);
    box-shadow:0 10px 24px rgba(37,99,235,.28)}}
  .brandmark svg{{width:22px;height:22px}}
  h1{{font-size:18px;line-height:1.2;margin:0;letter-spacing:-.025em}}
  .subtitle{{font-size:11.5px;color:var(--muted);margin:2px 0 0}}
  .header-actions{{display:flex;align-items:center;gap:9px}}
  .iconbtn{{appearance:none;width:36px;height:36px;display:grid;place-items:center;
    border:1px solid var(--border);border-radius:11px;background:var(--surface-1);
    color:var(--text-secondary);cursor:pointer;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
  .iconbtn:hover{{color:var(--series-1);border-color:var(--series-1)}}
  .iconbtn svg{{width:17px;height:17px}}
  .iconbtn.spinning svg{{animation:spin .65s ease}}
  @keyframes spin{{to{{transform:rotate(360deg)}}}}
  .mode{{display:inline-flex;align-items:center;gap:7px;font-size:10.5px;font-weight:700;
    letter-spacing:.11em;text-transform:uppercase;border:1px solid var(--border);
    border-radius:999px;padding:7px 11px;color:var(--text-secondary);background:var(--surface-1)}}
  .mode::before{{content:"";width:6px;height:6px;border-radius:50%;background:var(--good);
    box-shadow:0 0 0 3px var(--good-soft)}}
  .mode.live{{color:var(--critical);border-color:var(--critical)}}
  .mode.live::before{{background:var(--critical);box-shadow:0 0 0 3px var(--critical-soft)}}
  h2{{font-size:17px;margin:34px 0 12px;color:var(--text-primary);font-weight:680;
    letter-spacing:-.02em}}
  .sitenav{{margin:-10px 0 20px;padding-left:50px}}
  .sitenav a{{display:inline-flex;align-items:center;gap:5px;font-size:12px;
    font-weight:600;color:var(--muted);text-decoration:none}}
  .sitenav a:hover,.sitenav a:focus-visible{{color:var(--series-1)}}
  .banner{{border-radius:14px;padding:13px 16px;margin-bottom:12px;font-size:13px;
    line-height:1.55;border:1px solid var(--border);background:var(--surface-1);box-shadow:var(--shadow)}}
  .banner.critical{{border-color:rgba(210,63,76,.28);background:var(--critical-soft)}}
  .banner.warning{{border-color:rgba(170,104,0,.26);background:var(--warning-soft)}}
  .banner.info{{border-color:rgba(37,99,235,.22);background:var(--series-soft)}}
  .tiles{{display:grid;gap:12px;grid-template-columns:repeat(3,minmax(0,1fr))}}
  .tile{{min-width:0;background:linear-gradient(145deg,var(--surface-1),var(--surface-2));
    border:1px solid var(--border);border-radius:17px;padding:18px 19px;box-shadow:var(--shadow)}}
  .tl{{color:var(--text-secondary);font-size:11.5px;font-weight:600}}
  .tv{{font-size:clamp(22px,3vw,28px);line-height:1.16;font-weight:700;margin:7px 0 6px;
    letter-spacing:-.035em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
  .ts{{color:var(--muted);font-size:11px;line-height:1.45}}
  .tile.good .tv{{color:var(--good)}} .tile.bad .tv,.tile.critical .tv{{color:var(--critical)}}
  .tile.warning .tv{{color:var(--warning)}}
  .card{{background:var(--surface-1);border:1px solid var(--border);
    border-radius:18px;padding:18px;position:relative;box-shadow:var(--shadow)}}
  .card.empty{{color:var(--muted);font-size:13.5px}}
  .chart{{width:100%;height:270px;display:block;overflow:visible}}
  .tick{{fill:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}}
  .tip{{position:absolute;pointer-events:none;opacity:0;background:var(--surface-1);
    border:1px solid var(--border);border-radius:10px;padding:7px 10px;font-size:11.5px;
    box-shadow:0 10px 28px rgba(0,0,0,.18);transition:opacity .1s;white-space:nowrap}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{text-align:left;color:var(--muted);font-weight:650;text-transform:uppercase;
    letter-spacing:.055em;font-size:9.5px;border-bottom:1px solid var(--border);
    padding:5px 10px 11px;white-space:nowrap}}
  td{{padding:13px 10px;border-bottom:1px solid var(--gridline)}}
  tr:last-child td{{border-bottom:none}}
  tbody tr{{transition:background .15s ease}}
  tbody tr:hover{{background:var(--surface-2)}}
  .n{{text-align:right;font-variant-numeric:tabular-nums}}
  .pos{{color:var(--good)}} .neg{{color:var(--critical)}}
  .scroll{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
  .events{{list-style:none;margin:0;padding:0}}
  .ev{{padding:12px 2px;border-bottom:1px solid var(--gridline);font-size:13px}}
  .ev:last-child{{border-bottom:none}}
  .ev.quiet{{color:var(--muted)}}
  .tag{{display:inline-flex;align-items:center;font-size:9.5px;font-weight:700;text-transform:uppercase;
    letter-spacing:.07em;padding:3px 7px;border-radius:999px;margin-right:7px;
    border:1px solid var(--border);color:var(--text-secondary);background:var(--surface-2)}}
  .ev.ok .tag{{color:var(--good);border-color:transparent;background:var(--good-soft)}}
  .ev.block .tag{{color:var(--critical);border-color:transparent;background:var(--critical-soft)}}
  .ev.sold .tag{{color:var(--warning);border-color:transparent;background:var(--warning-soft)}}
  .ev.fixed .tag{{color:var(--good);border-color:transparent;background:var(--good-soft)}}
  .verdict{{display:flex;gap:14px;align-items:flex-start;margin:0 0 12px;
    padding:18px;border-radius:18px;font-size:14px;line-height:1.5;
    border:1px solid var(--border);background:var(--surface-1);box-shadow:var(--shadow)}}
  .verdict .dot{{position:relative;width:11px;height:11px;border-radius:50%;margin-top:5px;flex:0 0 11px}}
  .verdict .dot::after{{content:"";position:absolute;inset:-5px;border-radius:50%;opacity:.18;background:inherit}}
  .verdict.ok .dot{{background:var(--good)}}
  .verdict.warning .dot{{background:var(--warning)}}
  .verdict.critical .dot{{background:var(--critical)}}
  .verdict.ok{{border-color:rgba(7,131,82,.25)}}
  .verdict.warning{{border-color:rgba(170,104,0,.28)}}
  .verdict.critical{{border-color:rgba(210,63,76,.3)}}
  .verdict strong{{font-size:15px;margin-right:3px}}
  .vsub{{display:block;color:var(--muted);font-size:11.5px;margin-top:3px}}
  .copybtn{{appearance:none;border:0;background:var(--series-1);color:white;font:inherit;
    font-size:12.5px;font-weight:650;padding:9px 14px;border-radius:10px;cursor:pointer;
    box-shadow:0 8px 20px var(--series-soft)}}
  .copybtn:hover{{filter:brightness(1.08)}}
  .copied{{margin-left:9px;color:var(--good);font-size:12.5px}}
  pre{{white-space:pre-wrap;word-break:break-word;font-size:11.5px;
    line-height:1.55;background:var(--surface-2);border:1px solid var(--gridline);
    border-radius:11px;padding:12px;margin:10px 0 0;overflow-x:auto}}
  .tabs{{position:sticky;top:10px;z-index:20;display:flex;gap:4px;margin:18px 0 24px;
    padding:5px;border:1px solid var(--border);border-radius:14px;background:rgba(13,19,29,.82);
    box-shadow:var(--shadow);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px)}}
  @media (prefers-color-scheme:light){{.tabs{{background:rgba(255,255,255,.86)}}}}
  .tab{{appearance:none;flex:1;display:flex;align-items:center;justify-content:center;gap:7px;
    background:none;border:none;color:var(--muted);font:inherit;font-size:12px;
    font-weight:620;padding:9px 12px;cursor:pointer;border-radius:10px}}
  .tab svg{{width:15px;height:15px;stroke-width:1.8}}
  .tab:hover{{color:var(--text-primary);background:var(--surface-2)}}
  .tab[aria-selected="true"]{{color:var(--series-1);background:var(--series-soft)}}
  .tab:focus-visible{{outline:2px solid var(--series-1);outline-offset:1px}}
  .panel h2:first-child{{margin-top:0}}
  h3{{font-size:14px;margin:24px 0 9px;color:var(--text-primary)}}
  .note{{color:var(--muted);font-size:12.5px;line-height:1.6;margin:10px 0 0;
    max-width:70ch}}
  .why{{color:var(--text-secondary);font-size:12.5px;line-height:1.5}}
  .warn-inline{{color:var(--warning);font-size:11.5px}}
  .muted-inline{{color:var(--muted);font-size:11.5px}}
  .brief p{{margin:0 0 10px}} .brief p:last-child{{margin:0}}
  footer{{margin-top:42px;padding-top:20px;border-top:1px solid var(--gridline);
    color:var(--muted);font-size:11px;line-height:1.7}}
  @media (max-width:700px) {{
    body::before{{height:260px;background:radial-gradient(520px 230px at 45% -70px,var(--series-soft),transparent 75%)}}
    .wrap{{padding:22px 16px calc(var(--nav-height) + 38px)}}
    header{{margin-bottom:21px}}
    .brandmark{{width:36px;height:36px;border-radius:11px}}
    .subtitle{{display:none}}
    .sitenav{{padding-left:48px;margin-top:-15px}}
    .tiles{{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}
    .tile{{padding:15px 14px;border-radius:15px}}
    .tv{{font-size:clamp(20px,6.1vw,27px)}}
    .ts{{font-size:10.5px}}
    .chart{{height:220px}}
    .card{{padding:14px;border-radius:16px}}
    h2{{font-size:16px;margin-top:30px}}
    /* Nine columns do not fit on a phone. Hide the reference ones so the
       answer -- protected, how far from the stop, up or down -- is on screen
       without a sideways scroll nobody knows is there. */
    .positions-detail .sec{{display:none}}
    .tabs{{position:fixed;left:10px;right:10px;bottom:10px;top:auto;margin:0;
      height:var(--nav-height);padding:5px;z-index:50;border-radius:18px;
      padding-bottom:max(5px,env(safe-area-inset-bottom))}}
    .tab{{min-width:0;flex-direction:column;gap:1px;padding:6px 2px;font-size:9px;line-height:1.1}}
    .tab svg{{width:18px;height:18px}}
    .positions-summary th:nth-child(2),.positions-summary td:nth-child(2),
    .positions-summary th:nth-child(3),.positions-summary td:nth-child(3){{display:none}}
    .positions-summary td,.positions-summary th{{padding-left:8px;padding-right:8px}}
    .ev{{font-size:12.5px}}
    .verdict{{padding:15px;border-radius:16px}}
    .banner{{padding:12px 14px;border-radius:14px}}
  }}
  @media (max-width:390px) {{
    .wrap{{padding-left:12px;padding-right:12px}}
    .mode{{padding:6px 9px}}
    .iconbtn{{display:none}}
    .tile{{padding:14px 12px}}
    .tl{{font-size:10.5px}}
    .tv{{font-size:20px}}
  }}
  @media (prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}*{{animation:none!important;transition:none!important}}}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <span class="brandmark" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 17l5-5 4 3 7-8"/><path d="M15 7h5v5"/>
        </svg>
      </span>
      <div>
        <h1>{_esc(title)}</h1>
        <p class="subtitle">Automated swing-trading monitor</p>
      </div>
    </div>
    <div class="header-actions">
      <button class="iconbtn" id="refreshpage" type="button" aria-label="Refresh dashboard" title="Refresh dashboard">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M20 6v5h-5"/><path d="M4 18v-5h5"/><path d="M18.5 9A7 7 0 0 0 6.2 6.2L4 11M20 13l-2.2 4.8A7 7 0 0 1 5.5 15"/>
        </svg>
      </button>
      <span class="mode {'live' if mode == 'LIVE' else ''}">{_esc(mode)}</span>
    </div>
  </header>

  <nav class="sitenav"><a href="./floor.html">See the execution flow <span aria-hidden="true">&rarr;</span></a></nav>

  <!-- One glance answers "is anything wrong". Sits above everything, because
       the answer has to be readable without scrolling or interpreting. -->
  {verdict_block}

  <!-- Banners sit OUTSIDE the tabs on purpose. An unprotected position or a
       dead agent is not something to go looking for under a heading. -->
  <div id="banners">{banner}</div>

  <div class="tabs" role="tablist">
    <button class="tab" role="tab" data-panel="overview" aria-selected="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      <span>Overview</span>
    </button>
    <button class="tab" role="tab" data-panel="positions" aria-selected="false">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><path d="M4 19V9"/><path d="M10 19V5"/><path d="M16 19v-7"/><path d="M22 19H2"/></svg>
      <span>Positions</span>
    </button>
    <button class="tab" role="tab" data-panel="performance" aria-selected="false">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><path d="M3 17l6-6 4 4 8-9"/><path d="M15 6h6v6"/></svg>
      <span>Performance</span>
    </button>
    <button class="tab" role="tab" data-panel="rules" aria-selected="false">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
      <span>How it works</span>
    </button>
  </div>

  <section class="panel" data-panel="overview">
    <div class="tiles">{tiles}</div>

    <h2>Account equity</h2>
    {chart_block}

    <h2>Open positions</h2>
    {pos_block}

    <h2>What the agent did on its last run</h2>
    {today_block}

    <h2>Briefing</h2>
    {brief_block}

    <h2>Something wrong?</h2>
    <div class="card">
      <p class="note" style="margin-top:0">Copy this and send it to Claude. It
      carries the timestamps, errors, checks and positions that are needed to
      work out what happened &mdash; including the details nobody thinks to
      mention.</p>
      <button class="copybtn" id="copyreport">Copy report for Claude</button>
      <span class="copied" id="copied" hidden>Copied</span>
      <details style="margin-top:10px">
        <summary class="note" style="cursor:pointer">See what gets copied</summary>
        <pre id="reporttext">{report_esc}</pre>
      </details>
    </div>
  </section>

  <section class="panel" data-panel="positions" hidden>
    <h2>What you are holding</h2>
    {detail_block}
  </section>

  <section class="panel" data-panel="performance" hidden>
    <h2>Closed trades</h2>
    {perf_block}

    <h2>Recently closed</h2>
    {trades_block}
  </section>

  <section class="panel" data-panel="rules" hidden>
    <h2>How this agent decides</h2>
    {rules_block}
  </section>

  <footer>
    {_esc(stamp)}.
    Numbers update only when the agent runs; this page does not poll anything.<br>
    A paper account is fake money. Results here are not a prediction of what
    real money would do, and past results do not tell you what comes next.
  </footer>
</div>
<script>
/* ---------------------------------------------------------------------------
   Freshness, computed in the browser rather than baked into the page.

   The age of the last run was previously written into the HTML by Python at
   the moment the page was generated, so it read "0 min ago" forever after.
   That is the single most misleading thing a monitoring page can do: it made
   a day-old snapshot look live. Both problems are fixed here.

   1. The age is recomputed from a fixed timestamp every time the page is
      opened, and every minute it stays open.
   2. A tiny data.json is fetched with caching disabled. If the agent has run
      since this copy of the page was built, the browser is sent to a URL it
      has never seen, which is the only reliable way past an iOS home-screen
      cache.
--------------------------------------------------------------------------- */
(function () {{
  var BUILT = "{built_iso}";
  var DATED = "{dated_iso}";

  function ageParts(iso) {{
    if (!iso) return {{ text: "never run", hrs: 1e9, sev: "warning" }};
    var t = Date.parse(iso);
    if (isNaN(t)) return {{ text: "unknown", hrs: 1e9, sev: "critical" }};
    var hrs = (Date.now() - t) / 3600000;
    if (hrs < 0) hrs = 0;
    var hours = Math.round(hrs);
    var text = hrs < 1 ? Math.round(hrs * 60) + " min ago"
             : hrs < 48 ? hours + (hours === 1 ? " hour ago" : " hours ago")
             : Math.round(hrs / 24) + " days ago";
    var sev = hrs < 30 ? "good" : (hrs < 96 ? "warning" : "critical");
    return {{ text: text, hrs: hrs, sev: sev }};
  }}

  function paint() {{
    var a = ageParts(DATED);
    var val = document.getElementById("age-val");
    var tile = document.getElementById("age-tile");
    if (val) val.textContent = a.text;
    if (tile) tile.className = "tile " + a.sev;

    var host = document.getElementById("banners");
    if (!host) return;
    var old = document.getElementById("stale-banner");
    if (old) old.parentNode.removeChild(old);
    if (a.hrs > 30 && DATED) {{
      var b = document.createElement("div");
      b.id = "stale-banner";
      b.className = "banner " + (a.hrs > 96 ? "critical" : "warning");
      b.innerHTML = (a.hrs > 96
        ? "<strong>Stale.</strong> The agent last ran " + a.text +
          ". Nothing on this page is current. Check that the scheduled run is still working."
        : "<strong>Getting old.</strong> Last run " + a.text +
          ". Fine over a weekend or a market holiday, not fine on a Wednesday.");
      host.insertBefore(b, host.firstChild);
    }}
  }}

  paint();
  setInterval(paint, 60000);

  function checkForNewer() {{
    fetch("./data.json?t=" + Date.now(), {{ cache: "no-store" }})
      .then(function (r) {{ return r.ok ? r.json() : null; }})
      .then(function (d) {{
        if (d && d.updated_at && d.updated_at !== BUILT) {{
          location.replace("./index.html?v=" + encodeURIComponent(d.updated_at));
        }}
      }})
      .catch(function () {{ /* opened from a file, or offline: keep what we have */ }});
  }}

  checkForNewer();
  // Coming back to a home-screen app does not reload it, so check again.
  document.addEventListener("visibilitychange", function () {{
    if (!document.hidden) {{ paint(); checkForNewer(); }}
  }});
}})();

/* ---------------------------------------------------------------------------
   The report, copied in one click. The loop this serves: something breaks,
   the page says so, and the whole picture reaches Claude in one paste rather
   than being described from memory.
--------------------------------------------------------------------------- */
(function () {{
  var btn = document.getElementById('copyreport');
  var pre = document.getElementById('reporttext');
  var ok = document.getElementById('copied');
  if (!btn || !pre) return;

  function flash(msg) {{
    if (!ok) return;
    ok.textContent = msg;
    ok.hidden = false;
    setTimeout(function () {{ ok.hidden = true; }}, 4000);
  }}

  // The clipboard API needs a secure context and a permission that some
  // browsers refuse. Selecting the text is not as good as copying it, but it
  // is one keystroke away rather than a dead button.
  function select() {{
    var d = pre.closest('details');
    if (d) d.open = true;
    try {{
      var r = document.createRange();
      r.selectNodeContents(pre);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(r);
      flash('Selected \u2014 press Cmd/Ctrl+C');
    }} catch (e) {{
      flash('Open the details below and copy it');
    }}
  }}

  btn.addEventListener('click', function () {{
    var text = pre.textContent;
    if (navigator.clipboard && navigator.clipboard.writeText) {{
      navigator.clipboard.writeText(text).then(
        function () {{ flash('Copied'); }},
        select);
    }} else {{
      select();
    }}
  }});
}})();

/* Manual refresh is useful when the dashboard is running from a phone's home
   screen, where returning to the app does not always trigger a navigation. */
(function () {{
  var btn = document.getElementById('refreshpage');
  if (!btn) return;
  btn.addEventListener('click', function () {{
    btn.classList.add('spinning');
    btn.setAttribute('aria-label', 'Refreshing dashboard');
    location.reload();
  }});
}})();

/* ---------------------------------------------------------------------------
   Tabs. The panel lives in the URL hash so a view can be linked to and
   survives the reload the freshness check performs -- landing someone back on
   Overview every time the agent runs would make the other tabs unusable.
   With no JS every panel is simply visible, which is worse-looking but still
   complete; nothing here is the only way to reach anything.
--------------------------------------------------------------------------- */
(function () {{
  var tabs = [].slice.call(document.querySelectorAll('.tab'));
  var panels = [].slice.call(document.querySelectorAll('.panel'));
  if (!tabs.length) return;

  function show(name, push) {{
    var found = false;
    panels.forEach(function (p) {{
      var mine = p.getAttribute('data-panel') === name;
      p.hidden = !mine;
      if (mine) found = true;
    }});
    if (!found) return show('overview', push);
    tabs.forEach(function (t) {{
      t.setAttribute('aria-selected',
        t.getAttribute('data-panel') === name ? 'true' : 'false');
    }});
    if (push && window.history && history.replaceState) {{
      history.replaceState(null, '', '#' + name);
    }}
  }}

  tabs.forEach(function (t) {{
    t.addEventListener('click', function () {{
      show(t.getAttribute('data-panel'), true);
    }});
    t.addEventListener('keydown', function (e) {{
      var i = tabs.indexOf(t);
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {{
        e.preventDefault();
        var n = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
        n.focus();
        show(n.getAttribute('data-panel'), true);
      }}
    }});
  }});

  window.addEventListener('hashchange', function () {{
    show((location.hash || '#overview').slice(1), false);
  }});
  show((location.hash || '#overview').slice(1), false);
}})();

(function(){{
  var pts = {pts};
  var card = document.getElementById('c1'), tip = document.getElementById('t1');
  if (!card || !pts.length) return;
  var svg = card.querySelector('svg');
  var cross = svg.querySelector('.crosshair'), dot = svg.querySelector('.cursor-dot');
  function at(clientX) {{
    var r = svg.getBoundingClientRect();
    var vx = (clientX - r.left) / r.width * 920;
    var best = 0, bd = Infinity;
    for (var i = 0; i < pts.length; i++) {{
      var d = Math.abs(pts[i].x - vx);
      if (d < bd) {{ bd = d; best = i; }}
    }}
    var p = pts[best];
    cross.setAttribute('x1', p.x); cross.setAttribute('x2', p.x);
    cross.setAttribute('opacity', '1');
    dot.setAttribute('cx', p.x); dot.setAttribute('cy', p.y);
    dot.setAttribute('opacity', '1');
    tip.innerHTML = p.d + ' &middot; <b>$' +
      p.v.toLocaleString(undefined, {{maximumFractionDigits: 0}}) + '</b>';
    tip.style.opacity = '1';
    tip.style.left = Math.min(Math.max(p.x / 920 * r.width - 48, 4), r.width - 140) + 'px';
    tip.style.top = Math.max(p.y / 280 * r.height - 40, 2) + 'px';
  }}
  card.addEventListener('mousemove', function(e){{ at(e.clientX); }});
  card.addEventListener('touchstart', function(e){{ at(e.touches[0].clientX); }}, {{passive:true}});
  card.addEventListener('touchmove', function(e){{ at(e.touches[0].clientX); }}, {{passive:true}});
  function hide(){{ tip.style.opacity='0'; cross.setAttribute('opacity','0');
    dot.setAttribute('opacity','0'); }}
  card.addEventListener('mouseleave', hide);
  card.addEventListener('touchend', hide);
}})();
</script>
</body>
</html>"""


def write_dashboard(title: str = "Trading agent") -> Path:
    path = SITE / "index.html"
    path.write_text(build_html(title=title))

    # A tiny companion file the page can poll without downloading itself.
    # This is what lets a phone notice a new run has happened even when it is
    # showing a cached copy of the page.
    st = load_state()
    (SITE / "data.json").write_text(json.dumps({
        "updated_at": st.get("updated_at"),
        "carried_from": st.get("carried_from"),
        # So the freshness of the figures and the freshness of the agent can be
        # told apart without parsing the page.
        "last_full_run": st.get("last_full_run"),
        "refresh_only": bool(st.get("refresh_only")),
        "mode": st.get("mode"),
        "equity": (st.get("account") or {}).get("equity"),
        "positions": len(st.get("positions") or []),
        "orders": len(st.get("orders") or []),
        # So the hub can show a badge without fetching the larger file below.
        "pending": len(st.get("pending") or []),
    }, indent=2))

    _write_hub_files(st)
    return path


# ---------------------------------------------------------------------------
# What the hub reads
# ---------------------------------------------------------------------------
#
# GitHub Pages is static, so the hub cannot query the agent -- it can only read
# files the agent published. These two are that interface.
#
# The payload is an explicit list of fields rather than the state dict dumped
# whole. A future run adding an internal field to state would otherwise publish
# it to a public URL the moment it was written, which is a bad default for a
# file that holds account figures.

HUB_FIELDS = (
    "updated_at", "last_full_run", "mode", "healthy", "carried_from",
    "carried_reason", "positions", "signals", "orders", "exits", "protected",
    "vetoes", "findings", "errors", "briefing", "recent_trades", "research",
    "strategy_by_symbol", "pending", "pending_expires_at", "config_notes",
    "approved_submitted", "approved_refused",
)


# Fields the hub treats as lists. A state file written before these existed
# has no key for them, and `null` where a list belongs makes every reader on
# the other side write the same defensive check.
HUB_LISTS = frozenset((
    "positions", "signals", "orders", "exits", "protected", "vetoes",
    "findings", "errors", "recent_trades", "pending", "config_notes",
    "approved_submitted", "approved_refused",
))


def _write_hub_files(st: dict) -> None:
    payload = {}
    for k in HUB_FIELDS:
        v = st.get(k)
        payload[k] = ([] if v is None else v) if k in HUB_LISTS else v

    acct = st.get("account") or {}
    payload["account"] = {k: acct.get(k) for k in
                          ("equity", "cash", "buying_power", "status",
                           "trading_blocked", "mode")}

    # 35 lines of "skipped, not in an uptrend" is noise on a roster. The hub
    # shows the count and the reasons, not every symbol.
    skipped = st.get("skipped") or []
    reasons = {}
    for item in skipped:
        r = (item.get("reason") or "").split(",")[0][:60]
        reasons[r] = reasons.get(r, 0) + 1
    payload["skipped_count"] = len(skipped)
    payload["skipped_reasons"] = sorted(
        ({"reason": r, "count": c} for r, c in reasons.items()),
        key=lambda x: -x["count"])

    (SITE / "agent.json").write_text(json.dumps(payload, indent=2, default=str))

    # The settings form is generated from this, so the hub cannot offer a field
    # the agent would refuse. Current values travel with it, so the form opens
    # showing what is actually in force rather than the defaults.
    try:
        from . import overrides
        from .config import AgentConfig

        cfg, notes = overrides.apply(AgentConfig())
        current = {}
        for key in overrides.EDITABLE:
            section, field = key.split(".", 1)
            current[key] = getattr(getattr(cfg, section), field)

        (SITE / "schema.json").write_text(json.dumps({
            "schema": overrides.schema(),
            "current": current,
            "overrides": overrides.load_raw(),
            "notes": notes,
            "watchlist_size": len(cfg.watchlist),
        }, indent=2, default=str))
    except Exception as exc:
        # A broken schema file must never stop the dashboard being published.
        (SITE / "schema.json").write_text(
            json.dumps({"error": f"could not build the schema: {exc}"}, indent=2))
