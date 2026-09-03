# Trend pullback trading agent

An automated implementation of the swing setup you already trade by hand:
50/200 SMA trend filter, pull back to the 20 EMA, buy the reclaim, stop under
the swing low, target at 2R.

It scans a watchlist every day, applies the rules with no discretion, sizes
each trade so a stop-out costs a fixed percentage of the account, and either
hands you a trade card to act on yourself or sends bracket orders to a paper
account.

## Read this part first

**This is not a money printer, and building it does not mean it works.** The
rules in here are a widely traded, publicly known pattern. If it had a large
persistent edge, it would have been arbitraged away. What automation actually
buys you is consistency: the bot will not skip a valid setup because it is
nervous, will not double size because it is angry about the last loss, and
will not move a stop. Those three habits cost most discretionary traders more
than any indicator ever gained them.

**The backtest is the weakest evidence in the system.** It is one strategy on
one watchlist over one stretch of history that will not repeat. Every rule you
add makes the backtest look better, because you are fitting it to the past.
The right order is: backtest, then paper trade for at least a few months, then
consider small real size. Skipping the middle step is the most expensive
mistake available here.

**Automation does not reduce risk. It executes risk faster.** A bug in a
sizing function is not a wrong answer on a homework problem, it is an order.
That is why live trading requires three separate switches and why the default
mode sends nothing anywhere.

## Why Alpaca and not Robinhood

Robinhood has no official API for stocks or options. Its only public API is
for crypto. There are unofficial Python libraries that automate Robinhood by
impersonating the mobile app, and they work until they do not: this violates
Robinhood's terms of service and accounts have been restricted for it. Do not
risk your brokerage account to skip a signup form.

Alpaca is a US broker with a documented API and a free paper-trading account
that behaves like the real thing. You can run this for a year without funding
it. If you eventually want the agent to trade real money, you would fund
Alpaca, and Robinhood stays where you trade by hand.

TradingView is not an execution venue either. It fires alerts, including
webhook alerts that POST to a server you control, which is a good way to feed
signals into a system like this later. It cannot place an order.

## Setup (macOS)

```bash
cd ~/Documents/trading-agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Confirm the engine's math is correct
python agent.py selftest

# 2. Test the rules on real history (downloads data, takes a minute)
python agent.py backtest --start 2018-01-01
open reports/backtest.html
```

If you skip the virtual environment, add `--break-system-packages` to the pip
command. The venv is cleaner.

## Which strategy should you actually run

```bash
python agent.py compare --start 2005-01-01
open reports/comparison.html
```

This runs six approaches on identical data with identical costs and prints
them side by side. Buy and hold, a 200-day trend filter, dual momentum,
cross-sectional momentum, an equal-weight control, and your trend pullback
rules. Five of the six were published by other people years ago, which is the
point: a rule invented last week and tuned until the backtest looks good tells
you nothing, because it was fitted to the same history you are testing on.

The output has three tables. Read the **out of sample** one first. It covers
the years after the split date, which is history none of these strategies were
designed against. A strategy that tops the full period but not that table was
fitted to the past, and you should not trade it.

Two things to hold onto while reading:

The winner is partly the luckiest, not purely the best. Test six strategies
and the top one beat five others by chance as well as by merit. If two
strategies are within a percent or two of each other, you have not learned
which is better.

Max drawdown matters more than return. A strategy you abandon at the bottom
pays you whatever you earned before you quit. Look at the worst year in each
row and ask honestly whether you would still be running it in month four of
that.

## Daily use

```bash
source .venv/bin/activate

# Today's setups only. No broker, no orders. Writes reports/signals_<date>.md
python agent.py scan --refresh

# The full daily job, in dry-run: finds setups, screens them, shows what it
# would send. Nothing leaves your machine.
python agent.py run

# The same thing, actually placing the paper orders
python agent.py run --submit

# Rebuild the dashboard and open it
python agent.py dashboard --open

# Where the paper account stands
python agent.py status
```

Run these after the close, not during the day. The rules evaluate a completed
daily bar, so running at noon means judging a bar that is not finished yet.

To connect a paper account, copy `.env.example` to `.env` and paste in the
paper keys from app.alpaca.markets.

## The researcher

`python agent.py run` puts every setup the rules found through a screen before
any order goes out. Two layers, on purpose.

**The earnings check** is a plain calendar lookup, no AI involved. If a stock
reports earnings inside the next 10 days, the trade is refused. An earnings
report is a scheduled overnight coin flip that can gap 10% straight through
your stop, and your position sizing assumed that could not happen. This is the
highest-value filter in the system and it costs nothing.

**The news review** is the part that is actually AI. It reads recent headlines
and can block a trade for concrete, dated event risk: a pending merger vote, a
guidance cut issued this week, an accounting investigation, a trading halt. It
is instructed not to reason about whether the stock will go up, because it
cannot know that and neither can anyone else.

The model has exactly one power: it can turn a yes into a no. There is no code
path that lets it pick a trade, increase a position, move a stop, or override
a stop-out. That limit is enforced in code, not by asking it politely, and
`tests/test_logic.py` has checks that prove it. A malformed or hostile
response from the model is treated as "no opinion", never as an instruction.

If the research layer cannot run at all, the agent skips the trade instead of
taking it unchecked. Missing a trade costs nothing but opportunity.

This needs an Anthropic API key in `.env`, and runs a few cents a day at this
watchlist size. Without one, the earnings filter still works and the news
review is silently skipped.

## The dashboard

Every run writes what it saw and did to `state/`, then rebuilds a single HTML
page at `site/index.html`. Open it with `python agent.py dashboard --open`.
There is an example at `site/example-dashboard.html` filled with made-up
numbers so you can see the layout before your first real run.

The page shows equity, open positions, what the agent bought, what it blocked
and why, the written briefing, and recently closed trades. It reads only the
agent's own records, so it cannot show you a trade that did not happen.

The most important thing on it is the freshness banner at the top. If the
agent has not run in a few days, the page says so in red. A dashboard that
quietly shows stale numbers while you believe they are live is worse than no
dashboard at all.

## Publishing the dashboard

`.github/workflows/daily.yml` runs the agent on GitHub's computers every
weekday at 21:30 UTC and publishes the dashboard to a web address you can open
from anywhere. Your Mac does not need to be awake. It costs nothing.

Before you enable it, one decision. **GitHub Pages on a free account only
works from a public repository.** Your code being public is harmless. Your
account equity, positions and trade history being public is a different
question. Three ways to handle it:

1. Keep it public while the account is paper. It is fake money, so there is
   nothing real to leak. Switch approaches before you ever fund it.
2. GitHub Pro, about $4/month, which allows Pages from a private repo.
3. Publish to Cloudflare Pages instead and put Cloudflare Access in front of
   it. The free tier covers this and gates the site to your email address.

Your API keys are never in the repo either way. They live in GitHub's
encrypted secrets: `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_BASE_URL`,
and `ANTHROPIC_API_KEY`.

The workflow runs `tests/test_logic.py` before it is allowed to trade. If a
change breaks the engine, the run stops instead of placing orders.

## What the numbers mean

**R** is the unit that makes trades comparable. 1R is what you risked on that
trade. A +2R win on SPY and a +2R win on TSLA made you the same money, even
though the share counts were wildly different. Judge trades in R, not dollars.

**Expectancy** is your average R per trade. It is the only backtest number
that matters. +0.20R means that over many trades you average a fifth of your
risk per trade. Negative expectancy with a high win rate is common and is a
trap: winning often while losing big is how accounts die.

**Max drawdown** is how far the account fell from its high. This is the number
that decides whether you can actually run a strategy. A system with a 35%
drawdown is unrunnable by most people, because they quit at the bottom.

**Profit factor** is gross wins divided by gross losses. Under 1.0 loses money.

**Time in market** matters for comparison. A strategy that is invested 30% of
the time and returns less than buy-and-hold may still be better risk-adjusted,
because your money was exposed for far fewer days.

## How the rules work, exactly

Entry, evaluated on each completed daily bar:

1. Average dollar volume over 20 days is above $20M. Thin stocks have wide
   spreads, and the spread is a cost you pay on every trade.
2. Close is above the 200-day SMA, and the 50-day SMA is above the 200-day.
3. Yesterday closed at or below the 20 EMA (the pullback) and today closed
   above it (the reclaim).
4. Stop is the lowest low of the last 5 bars, minus 0.10 ATR.
5. The stop must sit between 0.5 and 3.0 ATR from price. Tighter than that is
   noise and gets hit at random. Wider means the trade is too volatile to size.

If all five hold, a market buy goes in **at the next session's open**, never at
the signal bar's close. You cannot trade a price you only learn about after
the bell, and a backtest that pretends otherwise is lying.

Exits, in order of precedence within a bar:

1. Gap down through the stop at the open, filled at the open. Worse than your
   stop. This is the risk position sizing cannot protect you from.
2. Gap up through the target at the open, filled at the open.
3. Stop touched intraday.
4. Target touched intraday.
5. Close below the 50 SMA, exit at the next open.
6. 40 bars held, exit at the next open.

If a bar touches both the stop and the target, the backtest assumes the stop
hit first. A daily bar does not record the order of events inside the day, so
every ambiguous case is read pessimistically.

## Position sizing

```
shares = (equity x 1%) / (entry - stop)
```

$10,000 account, 1% risk, $2 stop distance, buy 50 shares. Tighter stop, more
shares. The dollar loss on a stop-out is the same either way. That is the
entire point, and it is the part of the system that decides whether you
survive a losing streak.

Three caps override it: no position over 25% of equity, no more than 5 open at
once, and no margin. There is also a circuit breaker that stops new entries if
the account is 20% below its high water mark. The breaker does not restart
itself.

## Changing the rules

Everything that affects money lives in `tbot/config.py`. Nothing is buried in
the strategy code. Change a number there, re-run `selftest`, then re-run the
backtest and see what it did to expectancy and drawdown, not just to return.

A warning about that loop: if you tune parameters until the backtest looks
good, you have fit the strategy to history you already know. The honest test is
whether performance holds on a period you did not tune on. Backtest 2018-2022,
then run 2023-today without touching anything.

## Files

```
agent.py                     command line entry point
tbot/config.py               every number that affects money
tbot/data.py                 market data loading and caching
tbot/indicators.py           SMA, EMA, ATR, swing low
tbot/strategy.py             the entry rules
tbot/risk.py                 position sizing and portfolio limits
tbot/backtest.py             event-driven daily backtester
tbot/broker.py               Alpaca adapter with the three safety locks
tbot/research.py             earnings filter and the LLM veto
tbot/state.py                what each run recorded
tbot/dashboard.py            builds site/index.html
tbot/report.py               trade cards and the HTML backtest report
tests/test_logic.py          correctness checks for all of the above
.github/workflows/daily.yml  the scheduled run and site publish
state/                       run history, committed on purpose
site/                        the dashboard that gets published
```

`tests/test_logic.py` does not test whether the strategy makes money. It tests
whether the code does what it claims: no lookahead, correct sizing math, fills
at the right bar, losses close to the 1R that was intended. Run it after any
change.

## Things this does not do yet

- No options. Options add spreads, expiration, and Greeks, and a naive bot
  gets eaten alive by all three.
- No short side. The rules are long-only, so the agent sits out bear markets
  entirely rather than profiting from them.
- No intraday data. Everything is daily bars.
- No sector limits. It will happily hold five technology names and call that
  five positions, when really it is one bet.
- The strategy underperformed buying and holding SPY over 2018 to 2026 in
  backtest. Automation, an earnings filter and a news veto do not fix a thin
  edge. Finding one is still the open problem, and it is the real work.
