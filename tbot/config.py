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
    # Index / sector ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "SMH",
    # Mega caps
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "AMD", "NFLX", "COST", "JPM", "V", "UNH", "XOM", "LLY", "WMT",
    "CRM", "ORCL", "ADBE",
]


@dataclass
class StrategyConfig:
    """The trend pullback rules, expressed as numbers."""

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
