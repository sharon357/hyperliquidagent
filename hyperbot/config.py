"""Settings, loaded from environment variables (and a .env file if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

MAINNET_URL = "https://api.hyperliquid.xyz"
TESTNET_URL = "https://api.hyperliquid-testnet.xyz"
MODES = ("paper", "testnet", "mainnet")


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # --- where and how to trade
    mode: str = "paper"  # paper | testnet | mainnet
    private_key: str = field(default="", repr=False)
    account_address: str = ""
    coins: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    core_coins: list[str] = field(default_factory=list)  # the configured list; `coins` may grow in auto mode
    universe_mode: str = "fixed"  # fixed = only HL_COINS; auto = also scan the exchange for other liquid perps
    universe_extra: int = 5  # how many scanned coins to research besides the core ones
    min_volume_usd: float = 5_000_000.0  # a perp needs this much 24h volume to be tradable
    max_alt_position_pct: float = 150.0  # per-coin cap (% of equity) for coins outside the core list
    confirm_mainnet: bool = False

    # --- folders
    state_dir: Path = Path("state")
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")

    # --- costs (used by the backtester and the paper broker)
    taker_fee: float = 0.00045  # 0.045 % per side
    slippage_bps: float = 2.0
    live_slippage: float = 0.01  # max slippage tolerated on live market orders
    paper_equity: float = 10_000.0
    sim_impact: float = 1.0  # multiplier on the simulated market-impact estimate (0 = off)
    sim_impact_halflife_min: float = 30.0  # how fast our own price pressure fades

    # --- hard risk limits (enforced in code, the agent cannot change them)
    max_leverage: float = 3.0  # gross exposure / equity
    max_position_pct: float = 100.0  # one coin's notional as % of equity
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 20.0
    max_trades_per_day: int = 20
    min_order_usd: float = 10.0
    default_stop_loss_pct: float = 3.0
    max_stop_loss_pct: float = 10.0
    rebalance_band: float = 0.10  # ignore resizes smaller than 10 % of the position
    require_stop: bool = True  # every position must carry a stop-loss
    review_move_pct: float = 3.0  # extra manager review when profit or loss moves this many percent of equity since the last one (0 = off)
    default_trail_mult: float = 0.0  # trailing used when a trade does not set its own: this many times its stop distance (0 = none)
    ai_sets_limits: bool = False  # let the AI manager change the risk limits itself, inside fixed outer bounds
    exposure_multiplier: float = 1.0  # autopilot only: scales every playbook position (still capped by max_leverage)

    # --- LLM agent (any OpenAI-compatible API; DeepSeek by default)
    llm_api_key: str = field(default="", repr=False)
    llm_base_url: str = "https://api.deepseek.com"
    agent_model: str = "deepseek-v4-pro"
    agent_effort: str = "high"  # DeepSeek thinking effort: low | high | max ("" = do not send)
    agent_max_steps: int = 25
    agent_max_tokens: int = 32000  # per model call, reasoning included
    # LLM prices in USD per 1M tokens (defaults: deepseek-v4-pro peak rates; off-peak is half)
    llm_price_input: float = 1.32
    llm_price_input_cached: float = 0.044
    llm_price_output: float = 3.96

    # --- autopilot research
    research_days: float = 180.0
    research_intervals: list[str] = field(default_factory=lambda: ["1h", "4h"])

    @property
    def api_url(self) -> str:
        """Trading endpoint. Paper mode reads real mainnet prices."""
        return TESTNET_URL if self.mode == "testnet" else MAINNET_URL

    @property
    def is_live(self) -> bool:
        return self.mode in ("testnet", "mainnet")

    def ensure_dirs(self) -> None:
        for d in (self.state_dir, self.data_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


def _read_private_key() -> str:
    key = os.getenv("HL_PRIVATE_KEY", "").strip()
    key_file = os.getenv("HL_PRIVATE_KEY_FILE", "").strip()
    if not key and key_file and Path(key_file).is_file():
        key = Path(key_file).read_text(encoding="utf-8").strip()
    return key


def load_settings(env_file: str | None = None) -> Settings:
    load_dotenv(env_file or ".env", override=False)
    mode = os.getenv("HL_MODE", "paper").strip().lower() or "paper"
    if mode not in MODES:
        raise ValueError(f"HL_MODE must be one of {MODES}, got {mode!r}")
    coins = [c.strip().upper() for c in os.getenv("HL_COINS", "BTC,ETH,SOL").split(",") if c.strip()]
    d = Settings()
    universe_mode = os.getenv("HL_UNIVERSE", d.universe_mode).strip().lower() or d.universe_mode
    if universe_mode not in ("fixed", "auto"):
        raise ValueError("HL_UNIVERSE must be 'fixed' or 'auto'")
    s = Settings(
        core_coins=list(coins),
        universe_mode=universe_mode,
        universe_extra=_int("HL_UNIVERSE_SIZE", d.universe_extra),
        min_volume_usd=_float("HL_MIN_VOLUME_USD", d.min_volume_usd),
        max_alt_position_pct=_float("HL_MAX_ALT_POSITION_PCT", d.max_alt_position_pct),
        mode=mode,
        private_key=_read_private_key(),
        account_address=os.getenv("HL_ACCOUNT_ADDRESS", "").strip(),
        coins=coins,
        confirm_mainnet=_bool("HL_CONFIRM_MAINNET", False),
        # Real accounts keep their own risk state and journal, apart from the paper account's.
        state_dir=Path(os.getenv("HL_STATE_DIR", "state")) / ("" if mode == "paper" else mode),
        data_dir=Path(os.getenv("HL_DATA_DIR", "data")),
        reports_dir=Path(os.getenv("HL_REPORTS_DIR", "reports")),
        taker_fee=_float("HL_TAKER_FEE", d.taker_fee),
        slippage_bps=_float("HL_SLIPPAGE_BPS", d.slippage_bps),
        live_slippage=_float("HL_LIVE_SLIPPAGE", d.live_slippage),
        paper_equity=_float("HL_PAPER_EQUITY", d.paper_equity),
        sim_impact=_float("HL_SIM_IMPACT", d.sim_impact),
        sim_impact_halflife_min=_float("HL_SIM_IMPACT_HALFLIFE_MIN", d.sim_impact_halflife_min),
        max_leverage=_float("HL_MAX_LEVERAGE", d.max_leverage),
        max_position_pct=_float("HL_MAX_POSITION_PCT", d.max_position_pct),
        max_daily_loss_pct=_float("HL_MAX_DAILY_LOSS_PCT", d.max_daily_loss_pct),
        max_drawdown_pct=_float("HL_MAX_DRAWDOWN_PCT", d.max_drawdown_pct),
        max_trades_per_day=_int("HL_MAX_TRADES_PER_DAY", d.max_trades_per_day),
        min_order_usd=_float("HL_MIN_ORDER_USD", d.min_order_usd),
        default_stop_loss_pct=_float("HL_DEFAULT_STOP_LOSS_PCT", d.default_stop_loss_pct),
        max_stop_loss_pct=_float("HL_MAX_STOP_LOSS_PCT", d.max_stop_loss_pct),
        rebalance_band=_float("HL_REBALANCE_BAND", d.rebalance_band),
        require_stop=_bool("HL_REQUIRE_STOP", d.require_stop),
        default_trail_mult=_float("HL_DEFAULT_TRAIL_MULT", d.default_trail_mult),
        review_move_pct=_float("HL_REVIEW_MOVE_PCT", d.review_move_pct),
        ai_sets_limits=_bool("HL_AI_SETS_LIMITS", d.ai_sets_limits),
        exposure_multiplier=_float("HL_EXPOSURE_MULTIPLIER", d.exposure_multiplier),
        llm_api_key=(os.getenv("HL_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "").strip(),
        llm_base_url=os.getenv("HL_LLM_BASE_URL", d.llm_base_url).strip() or d.llm_base_url,
        agent_model=os.getenv("HL_AGENT_MODEL", d.agent_model).strip() or d.agent_model,
        agent_effort=os.getenv("HL_AGENT_EFFORT", d.agent_effort).strip(),
        agent_max_steps=_int("HL_AGENT_MAX_STEPS", d.agent_max_steps),
        agent_max_tokens=_int("HL_AGENT_MAX_TOKENS", d.agent_max_tokens),
        llm_price_input=_float("HL_LLM_PRICE_INPUT", d.llm_price_input),
        llm_price_input_cached=_float("HL_LLM_PRICE_INPUT_CACHED", d.llm_price_input_cached),
        llm_price_output=_float("HL_LLM_PRICE_OUTPUT", d.llm_price_output),
        research_days=_float("HL_RESEARCH_DAYS", d.research_days),
        research_intervals=[i.strip() for i in os.getenv("HL_RESEARCH_INTERVALS", "1h,4h").split(",") if i.strip()],
    )
    s.ensure_dirs()
    return s
