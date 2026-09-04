"""
Configuration for the trend pullback agent.

Every number that affects money is in this file. Nothing is hidden in the
strategy code. If you want to change how the agent behaves, change it here.
"""

from dataclasses import dataclass, field, asdict
from typing import List


# ---------------------------------------------------------------------------
# What the agent watches
# ---------------------------------------------------------------------------

# Liquid large caps and index ETFs. These are chosen because they have tight
# spreads, which matters more than most beginners expect: a wide spread eats
# your edge on every single trade, entry and exit.
DEFAULT_WATCHLIST: List[str] = [
    # Roughly 150 liquid US large caps, spread across sectors on purpose.
    #
    # Why this many: the entry rules only fire about 3 to 4 times a year on any
    # one stock, and the position cap almost never binds. Trade count therefore
    # tracks the length of this list, near enough to a straight line. Thirty
    # symbols produced about 105 trades a year. This produces roughly three
    # times that, which is what shortens the wait for a result that can be told
    # apart from luck.
    #
    # Why spread across sectors: the old list was mostly big tech, and the
    # correlation check exists precisely to refuse the same bet twice. A list
    # that is really one bet wastes slots on near-duplicates.
    #
    # An honest caveat that belongs next to the list itself. These are
    # companies that are large and liquid in 2026. Any backtest run over this
    # list is flattered by that, because the ones that failed on the way here
    # are not in it. Forward from today the list is clean, since nothing in it
    # was chosen with knowledge of what happens next. Backward, it is not
    # evidence. Do not read a backtest on this list as a forecast.

    # Index and sector ETFs
    "SPY", "QQQ", "IWM", "DIA", "MDY", "RSP",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
    "SMH", "IBB", "ITB", "XRT", "KRE", "GLD", "SLV", "TLT", "HYG", "EFA", "EEM",

    # Technology and semiconductors
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "QCOM", "TXN", "INTC", "MU", "AMAT",
    "LRCX", "KLAC", "ADI", "NXPI", "MRVL", "ON", "SNPS", "CDNS", "ANET", "SMCI",

    # Software and internet
    "GOOGL", "META", "AMZN", "NFLX", "CRM", "ORCL", "ADBE", "NOW", "INTU",
    "PANW", "SNOW", "DDOG", "WDAY", "TEAM", "SHOP", "UBER", "ABNB", "SPOT",

    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "SPGI", "AXP",
    "V", "MA", "PYPL", "COF", "USB", "PNC", "CB", "PGR", "AIG", "MET",

    # Healthcare
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "GILD", "BMY", "VRTX", "REGN", "ISRG", "SYK", "BSX", "MDT", "CI", "HCA",

    # Consumer
    "WMT", "COST", "HD", "LOW", "TGT", "NKE", "SBUX", "MCD", "CMG", "TJX",
    "PG", "KO", "PEP", "PM", "MDLZ", "CL", "KMB", "GIS", "DG", "ROST",

    # Industrials and transport
    "CAT", "DE", "HON", "GE", "BA", "LMT", "RTX", "NOC", "UNP", "CSX",
    "UPS", "FDX", "ETN", "EMR", "PH", "ITW", "MMM", "WM", "CARR", "JCI",

    # Energy, materials, utilities, real estate
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY", "WMB", "KMI",
    "LIN", "APD", "SHW", "FCX", "NEM", "NUE", "DOW",
    "NEE", "DUK", "SO", "D", "AEP", "EXC",
    "AMT", "PLD", "EQIX", "SPG", "O", "CCI",

    # Communications, autos and other large caps
    # EA was removed: it stopped trading on 4 August 2026 when the buyout
    # closed and it was delisted from NASDAQ. The watcher caught it on the
    # first live run, from a price feed that had been frozen for weeks, which
    # is exactly the failure that check exists for. Assume there are others in
    # this list: it was written from memory, and the only reliable way to find
    # a dead ticker is to watch for one.
    "TSLA", "GM", "F", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "BRK-B", "MMC", "AON", "ADP", "PAYX", "FI", "CTAS", "ORLY", "AZO", "YUM",
]


@dataclass
class StrategyConfig:
    """The trading rules, expressed as numbers.

    Three rule sets live here and any combination of them can be switched on.
    They look for genuinely different things, which is the point: a pullback
    buys weakness inside strength, a breakout buys strength itself, and mean
    reversion buys a stock everybody just sold. They rarely fire on the same
    day on the same stock, so running all three raises the trade count without
    taking the same trade twice.

    What does NOT change between them is the discipline. Every one of them
    sizes off the same 1% risk, places a stop before the order goes in, takes
    the same liquidity floor, and is judged by the same journal that refuses to
    call a small sample an edge.
    """

    # Which rule sets are live. Order matters only for tie-breaking: a symbol
    # produces at most one trade per day, from the first strategy that fires.
    enabled: List[str] = field(
        default_factory=lambda: ["pullback", "breakout", "reversion"])

    # --- Trend filter -------------------------------------------------------
    # Price must be above the long moving average and the medium average must
    # be above the long one. This is the "only trade with the tide" rule.
    sma_fast: int = 50
    sma_slow: int = 200

    # --- Pullback / entry ---------------------------------------------------
    # Price pulls back to the 20 EMA, then closes back above it. That reclaim
    # bar is the trigger. We enter at the NEXT session's open, never at the
    # close of the bar that produced the signal, because you cannot actually
    # trade a close you only learn about after it happened.
    ema_pullback: int = 20

    # --- Stop placement -----------------------------------------------------
    # Stop goes under the recent swing low, with a small ATR buffer so normal
    # noise does not knock you out.
    atr_period: int = 14
    swing_lookback: int = 5
    stop_atr_buffer: float = 0.10

    # Guard rails on stop distance, measured in ATR. A stop tighter than this
    # is noise, not a stop. A stop wider than this means the trade is too
    # volatile to size sensibly, so we skip it.
    min_stop_atr: float = 0.50
    max_stop_atr: float = 3.00

    # --- Target -------------------------------------------------------------
    # 2:1 reward to risk. Target = entry + 2 * (entry - stop).
    reward_risk: float = 2.0

    # --- Exits --------------------------------------------------------------
    # Bail if the trend itself breaks, and do not let a trade sit forever.
    exit_on_trend_break: bool = True   # close below the 50 SMA
    max_hold_days: int = 40

    # --- Choosing between candidates ----------------------------------------
    # When more setups appear than there are free slots, this decides which
    # ones get taken. "none" keeps the old behavior, which was list order and
    # therefore effectively arbitrary. On this agent's history 62% of setups
    # were discarded for want of a slot, so this setting has more influence on
    # results than most of the entry rules do.
    #   none         list order, arbitrary
    #   momentum     strongest 6-month return, skipping the last month
    #   reward_risk  tightest stop relative to the stock's own noise
    #   liquidity    heaviest dollar volume
    rank_by: str = "none"

    # --- Breakout rules -----------------------------------------------------
    # Same trend filter as the pullback, then: today's close is the highest
    # close of the last N sessions and yesterday's was not. That second half
    # matters. Without it every bar of a long run is a fresh "breakout" and
    # you buy the same move ten times.
    #
    # There is no swing low to hide behind at a new high, so the stop is a
    # straight multiple of the stock's own daily range.
    breakout_lookback: int = 50
    breakout_stop_atr: float = 2.0

    # --- Mean reversion rules -----------------------------------------------
    # Only in a long-term uptrend, only when the last two sessions have been
    # almost pure selling (2-period RSI under 10) and price is under its
    # 10-day average. This is buying a dip in something that is still healthy,
    # which is a different bet from the other two and fails in different
    # weather.
    #
    # It gets a wider stop because it enters while price is still falling, and
    # a much shorter leash because a bounce that has not happened within two
    # weeks is not going to.
    reversion_rsi_period: int = 2
    reversion_rsi_max: float = 10.0
    reversion_sma: int = 10
    reversion_stop_atr: float = 2.5
    reversion_max_hold: int = 10

    # --- Liquidity filter ---------------------------------------------------
    # Average dollar volume floor. Thin names have wide spreads and gap hard.
    min_avg_dollar_volume: float = 20_000_000.0
    dollar_volume_window: int = 20


@dataclass
class RiskConfig:
    """Position sizing and portfolio limits."""

    starting_equity: float = 10_000.0

    # Fraction of account equity risked per trade. 1% means a full stop-out
    # costs you 1% of the account, not 1% of the position. This is the single
    # most important number in the whole system.
    risk_per_trade: float = 0.01

    # No single position may exceed this share of equity, regardless of how
    # tight the stop is. A very tight stop would otherwise size you into a
    # position so large that one gap wipes out the account.
    max_position_pct: float = 0.25

    # No more than this many open trades at once.
    max_open_positions: int = 5

    # Refuse a new trade that moves almost identically to something you
    # already hold. Without this, an agent watching index ETFs will happily
    # buy DIA, QQQ and XLK on the same day, call it three positions, and take
    # three times the intended risk on what is really one bet. Correlation is
    # measured on daily returns, so it needs no sector labels and adapts when
    # things that used to move separately stop doing so.
    max_correlation: float = 0.80
    correlation_window: int = 60

    # Never use margin. Total cost basis of open positions stays under equity.
    max_gross_exposure: float = 1.00

    # Circuit breaker. If the account is this far below its high water mark,
    # the agent stops opening new trades and says so on the dashboard.
    max_drawdown_halt: float = 0.20

    # It starts taking trades again once the account has recovered to within
    # this much of its high water mark. Without a resume level the breaker is
    # a one-way switch that retires the strategy after a single bad stretch.
    resume_below: float = 0.10

    # And it resumes anyway after this many sessions, because a halted
    # strategy holds nothing, so its equity cannot recover, so recovery alone
    # would never arrive. Roughly three months.
    halt_cooldown_days: int = 60


@dataclass
class CostConfig:
    """Trading frictions. Ignoring these is how backtests lie to you."""

    # Slippage in basis points (1 bp = 0.01%) applied against you on entries
    # and on market exits.
    slippage_bps: float = 5.0

    # Extra slippage on stop fills, because stops become market orders in a
    # fast tape and fill worse than the trigger price.
    stop_slippage_bps: float = 10.0

    # Per-share commission. Zero at Alpaca and Robinhood for US equities.
    commission_per_share: float = 0.0

    # Flat per-order fee if your broker charges one.
    commission_per_order: float = 0.0


@dataclass
class ResearchConfig:
    """The research layer. It can only ever block a trade, never create one."""

    # Deterministic earnings check. Needs no API key and no AI. This is the
    # highest-value filter here: an earnings report is a scheduled overnight
    # coin flip that can gap straight through your stop.
    check_earnings: bool = True
    earnings_blackout_days: int = 10

    # LLM review of recent headlines. Needs ANTHROPIC_API_KEY. Costs roughly
    # a few cents a day at this watchlist size.
    use_llm: bool = True
    max_headlines: int = 8

    # If the research layer cannot run, skip the trade instead of taking it
    # unchecked. Set False and the agent will trade blind when the API is down.
    require_research: bool = True

    # Written summary at the end of each run.
    write_briefing: bool = True


@dataclass
class AgentConfig:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    watchlist: List[str] = field(default_factory=lambda: list(DEFAULT_WATCHLIST))

    # Safety switch. The agent will not send a live order unless this is
    # explicitly True AND the broker is pointed at a live endpoint AND you
    # passed --i-understand-the-risk on the command line. Three locks.
    allow_live_trading: bool = False

    def to_dict(self):
        return asdict(self)


DEFAULT = AgentConfig()
