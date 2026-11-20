"""One set of operations shared by the HTTP API and the MCP server.

Every trading operation goes through the same Executor and risk limits as the
CLI and the autopilot. On mainnet, trades are only placed when
HL_CONFIRM_MAINNET=yes; otherwise they are dry runs.
"""

from __future__ import annotations

import threading
from dataclasses import replace

from .auto import PortfolioAgent, PortfolioRunner
from .backtest import BacktestConfig, run_backtest
from .config import Settings, load_settings
from .data import HLData, load_history
from .exchange import PaperBroker, make_broker
from .risk import Executor, Journal, RiskManager
from .strategies import REGISTRY, get_strategy

# name -> (changes the account?, description, parameters)
OPERATIONS: dict[str, tuple[bool, str, dict[str, str]]] = {
    "account": (False, "Equity, open positions with stops, today's PnL, drawdown, halt state and the hard risk limits.", {}),
    "market": (False, "Live price, 24h change, funding, open interest, volume and order-book imbalance for every tradable coin.", {}),
    "signals": (False, "Indicators and every strategy's current signal for a coin on one timeframe.", {"coin": "e.g. BTC", "interval": "15m, 1h, 4h or 1d"}),
    "strategy_performance": (False, "Rank every strategy by a recent backtest on one coin (costs included).", {"coin": "e.g. BTC", "interval": "15m, 1h, 4h or 1d", "days": "7 to 180"}),
    "sentiment": (False, "Fear & Greed index, news headlines with a keyword score, and Hyperliquid positioning.", {}),
    "strategies": (False, "The built-in strategies, what they do and their default parameters.", {}),
    "backtest": (False, "Backtest one strategy on one coin.", {"coin": "e.g. BTC", "strategy": "a strategy name", "interval": "default 4h", "days": "default 180", "leverage": "default 1", "stop_loss_pct": "optional", "params": "optional object of strategy parameters"}),
    "research": (False, "The autopilot's scorecard: walk-forward (out-of-sample) and fitted results for every strategy, coin and timeframe.", {}),
    "playbook": (False, "The strategies the autopilot is currently running, with live signals.", {}),
    "journal": (False, "Recent decisions, trades and rejections.", {"count": "default 15, max 30"}),
    "simulation": (False, "Result so far of a `hyperbot simulate` run.", {"name": "default 'default'"}),
    "scan_market": (False, "Every liquid perp on the exchange with 24h change, volume, open interest and funding, and which coins the bot currently researches.", {}),
    "search_news": (False, "Search recent web news (7 days) for a query. Untrusted text.", {"query": "search terms"}),
    "trending": (False, "Coins retail is searching for most right now.", {}),
    "set_target": (True, "Trade so a coin holds target_usd of signed notional (positive long, negative short, 0 flat). The risk engine may shrink or reject it.", {"coin": "e.g. BTC", "target_usd": "signed USD", "stop_loss_pct": "percent from the current price", "take_profit_pct": "percent, 0 for none", "reason": "why"}),
    "close": (True, "Close a coin's whole position.", {"coin": "e.g. BTC", "reason": "why"}),
    "set_playbook": (True, "Replace the autopilot's playbook (it takes effect while `hyperbot auto` is running).", {"allocations": "list of {coin, interval, strategy, weight, stop_loss_pct}", "risk_scale": "0 to 1", "rationale": "why"}),
    "kill": (True, "Emergency stop: close every position and block new ones until `resume`.", {}),
    "resume": (True, "Clear a halt and restart drawdown tracking.", {}),
}


class _NoLLM:
    """Placeholder: the service uses the agent's tools, never its model."""


class Service:
    def __init__(
        self,
        settings: Settings | None = None,
        executor: Executor | None = None,
        runner: PortfolioRunner | None = None,
        lock: threading.Lock | None = None,
    ):
        """Pass `executor`, `runner` and `lock` to share one account with a running autopilot (see `hyperbot serve`)."""
        s = settings or load_settings()
        self.s = s
        self.dry_run = s.mode == "mainnet" and not s.confirm_mainnet
        self.executor = executor or Executor(s, make_broker(s), RiskManager(s), Journal(s.state_dir), dry_run=self.dry_run)
        self.dry_run = self.executor.dry_run
        self.data = HLData()
        self.lock = lock or threading.Lock()
        self.tools = PortfolioAgent(
            s, self.executor, self.data, client=_NoLLM(), runner=runner or PortfolioRunner(s, self.executor)
        )

    def describe(self) -> dict:
        return {
            "mode": self.s.mode,
            "dry_run": self.dry_run,
            "coins": self.s.coins,
            "operations": {
                name: {"changes_account": w, "description": d, "parameters": p} for name, (w, d, p) in OPERATIONS.items()
            },
        }

    def call(self, name: str, args: dict | None = None) -> dict:
        """Run an operation. Errors come back as {"error": ...} rather than raising."""
        args = dict(args or {})
        if name not in OPERATIONS:
            return {"error": f"unknown operation {name!r}", "available": sorted(OPERATIONS)}
        try:
            with self.lock:
                return getattr(self, f"_op_{name}")(args)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # ---- read-only
    def _op_account(self, a):
        return self.tools.tool_get_account(a)

    def _op_market(self, a):
        return self.tools.tool_get_market_overview(a)

    def _op_signals(self, a):
        return self.tools.tool_get_quant_signals({"coin": a["coin"], "interval": a.get("interval", "4h")})

    def _op_strategy_performance(self, a):
        return self.tools.tool_get_strategy_performance(
            {"coin": a["coin"], "interval": a.get("interval", "4h"), "days": a.get("days", 90)}
        )

    def _op_sentiment(self, a):
        return self.tools.tool_get_sentiment(a)

    def _op_strategies(self, a):
        return {n: {"description": c.description, "defaults": c.defaults} for n, c in REGISTRY.items()}

    def _op_backtest(self, a):
        coin = str(a["coin"]).upper()
        interval = a.get("interval", "4h")
        df = load_history(coin, interval, float(a.get("days", 180)), self.s.data_dir)
        strat = get_strategy(a["strategy"], **(a.get("params") or {}))
        sl = a.get("stop_loss_pct")
        cfg = BacktestConfig(
            leverage=float(a.get("leverage", 1.0)),
            taker_fee=self.s.taker_fee,
            slippage_bps=self.s.slippage_bps,
            stop_loss_pct=float(sl) if sl else None,
        )
        res = run_backtest(df, strat, cfg)
        return {"coin": coin, "interval": interval, "strategy": repr(strat), "from": str(df.index[0]), "to": str(df.index[-1]), "metrics": res.metrics}

    def _op_research(self, a):
        return self.tools.tool_get_research(a)

    def _op_playbook(self, a):
        return self.tools.tool_get_playbook(a)

    def _op_journal(self, a):
        return self.tools.tool_get_recent_decisions({"count": a.get("count", 15)})

    def _op_simulation(self, a):
        from .sim import SimTracker

        name = str(a.get("name", "default"))
        if not name.replace("-", "").replace("_", "").isalnum():
            return {"error": "simulation names may only contain letters, digits, - and _"}
        sim = replace(self.s, mode="paper", state_dir=self.s.state_dir / "sim" / name)
        if not sim.state_dir.is_dir():
            return {"error": f"no simulation named {name!r}; start one with `hyperbot simulate --name {name}`"}
        rep = SimTracker(sim, PaperBroker(sim, self.data)).report()
        return rep or {"error": "the simulation has not recorded anything yet"}

    def _op_scan_market(self, a):
        return self.tools.tool_scan_market(a)

    def _op_search_news(self, a):
        return self.tools.tool_search_news({"query": a.get("query", "")})

    def _op_trending(self, a):
        return self.tools.tool_get_trending(a)

    # ---- changes the account
    def _op_set_target(self, a):
        return self.executor.set_target(
            str(a["coin"]),
            float(a["target_usd"]),
            stop_loss_pct=float(a.get("stop_loss_pct") or 0) or None,
            take_profit_pct=float(a.get("take_profit_pct") or 0) or None,
            source="api",
            reason=str(a.get("reason", ""))[:500],
        )

    def _op_close(self, a):
        return self.executor.set_target(str(a["coin"]), 0.0, source="api", reason=str(a.get("reason", ""))[:500])

    def _op_set_playbook(self, a):
        return self.tools.tool_set_playbook(a)

    def _op_kill(self, a):
        self.executor.risk.halt("kill switch (api)")
        return {"halted": True, "closed": self.executor.flatten_all(source="api", reason="kill switch")}

    def _op_resume(self, a):
        self.executor.risk.resume(self.executor.broker.account().equity)
        return {"halted": False}
