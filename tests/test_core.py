import json

import numpy as np
import pandas as pd
import pytest

from hyperbot import indicators as ind
from hyperbot.backtest import BacktestConfig, grid_search, run_backtest, walk_forward
from hyperbot.config import Settings
from hyperbot.data import attach_funding
from hyperbot.exchange import AccountState, PaperBroker, Position
from hyperbot.risk import Executor, Journal, RiskManager
from hyperbot.signals import quant_snapshot
from hyperbot.strategies import REGISTRY, Strategy, get_strategy


def make_df(n=1500, seed=1, drift=0.0002, vol=0.01):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.003, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.003, n))
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1.0, "funding": 0.00001}, index=idx
    )


class Fixed(Strategy):
    """Test helper: a constant or scripted signal."""

    name = "fixed"

    def __init__(self, values):
        super().__init__()
        self.values = values

    def signal(self, df):
        if np.isscalar(self.values):
            return pd.Series(float(self.values), index=df.index)
        return pd.Series(self.values, index=df.index, dtype=float)


FREE = BacktestConfig(taker_fee=0.0, slippage_bps=0.0)


def test_indicators_basic():
    df = make_df()
    r = ind.rsi(df["close"]).dropna()
    assert r.between(0, 100).all()
    assert (ind.atr(df).dropna() > 0).all()
    assert set(ind.supertrend(df).unique()) <= {-1.0, 0.0, 1.0}


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_strategy_signals_in_range_and_causal(name):
    df = make_df()
    strat = get_strategy(name)
    full = strat.signal(df)
    assert len(full) == len(df)
    assert full.between(-1, 1).all()
    # No look-ahead: the signal on a truncated history must match the full run.
    cut = 1000
    part = strat.signal(df.iloc[:cut])
    assert np.allclose(part.to_numpy(), full.iloc[:cut].to_numpy(), equal_nan=True)


def test_buy_hold_matches_price_without_costs():
    df = make_df()
    df["funding"] = 0.0
    res = run_backtest(df, get_strategy("buy_hold"), FREE)
    # Enters at the open of bar 1 (signal from bar 0), holds to the last close.
    expected = df["close"].iloc[-1] / df["open"].iloc[1] - 1
    assert res.metrics["total_return_pct"] == pytest.approx(expected * 100, abs=0.02)


def test_costs_and_funding_reduce_equity():
    df = make_df()
    free = run_backtest(df.assign(funding=0.0), get_strategy("buy_hold"), FREE)
    costly = run_backtest(df, get_strategy("buy_hold"), BacktestConfig())
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert costly.metrics["fees_paid"] > 0
    assert costly.metrics["funding_paid"] > 0  # long pays positive funding


def test_short_profits_when_price_falls():
    df = make_df(drift=-0.001)
    df["funding"] = 0.0
    res = run_backtest(df, Fixed(-1.0), FREE)
    assert res.metrics["total_return_pct"] > 0


def test_signal_is_executed_on_the_next_bar():
    df = make_df(n=50)
    df["funding"] = 0.0
    values = np.zeros(50)
    values[10:] = 1.0  # decided at the close of bar 10
    res = run_backtest(df, Fixed(values), FREE)
    assert (res.exposure.iloc[:11] == 0).all()
    assert res.exposure.iloc[11] > 0
    assert res.trades.iloc[0]["entry_px"] == pytest.approx(df["open"].iloc[11])


def test_stop_loss_caps_the_loss_and_blocks_reentry():
    n = 60
    close = np.concatenate([np.full(10, 100.0), np.linspace(100, 60, 50)])
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    open_ = np.concatenate([[100.0], close[:-1]])
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close), "close": close, "volume": 1.0, "funding": 0.0}, index=idx)
    res = run_backtest(df, Fixed(1.0), replace_cfg(FREE, stop_loss_pct=5.0))
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["reason"] == "stop_loss"
    assert res.metrics["total_return_pct"] == pytest.approx(-5.0, abs=0.5)


def replace_cfg(cfg, **kw):
    from dataclasses import replace

    return replace(cfg, **kw)


def test_liquidation_at_high_leverage():
    n = 40
    close = np.concatenate([np.full(5, 100.0), np.linspace(100, 70, 35)])
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    open_ = np.concatenate([[100.0], close[:-1]])
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close), "close": close, "volume": 1.0, "funding": 0.0}, index=idx)
    res = run_backtest(df, Fixed(1.0), replace_cfg(FREE, leverage=10.0))
    assert res.metrics["liquidations"] == 1
    assert res.metrics["total_return_pct"] < -70
    assert res.equity.iloc[-1] >= 0


def test_funding_attaches_to_the_bar_it_settles():
    idx = pd.date_range("2025-01-01", periods=3, freq="4h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0}, index=idx)
    f = pd.Series(0.001, index=pd.date_range("2025-01-01 01:00", periods=8, freq="1h", tz="UTC"))
    out = attach_funding(df, f, "4h")
    assert out["funding"].tolist() == pytest.approx([0.004, 0.004, 0.0])


def test_grid_search_and_walk_forward_run():
    df = make_df(n=1200)
    ranked = grid_search(df, "ema_cross", BacktestConfig(), grid={"fast": [10, 20], "slow": [50, 100]})
    assert len(ranked) == 4
    wf = walk_forward(df, "ema_cross", BacktestConfig(), grid={"fast": [10, 20], "slow": [50, 100]}, n_splits=3)
    assert len(wf["folds"]) == 3
    assert isinstance(wf["folds"][0]["params"]["fast"], int)
    assert len(wf["oos_equity"]) == len(df) - len(df) // 4


def test_quant_snapshot_is_json_serialisable():
    snap = quant_snapshot(make_df(n=700))
    json.dumps(snap)
    assert set(snap["strategy_signals"]) == set(REGISTRY) - {"buy_hold"}


# ------------------------------------------------------------------ risk + execution
class FakeData:
    def __init__(self):
        self.prices = {"BTC": 100.0, "ETH": 10.0, "DOGE": 1.0}

    def mids(self):
        return dict(self.prices)


@pytest.fixture
def stack(tmp_path):
    s = Settings(state_dir=tmp_path, data_dir=tmp_path, reports_dir=tmp_path, coins=["BTC", "ETH"], paper_equity=10_000, slippage_bps=0.0, taker_fee=0.0)
    data = FakeData()
    broker = PaperBroker(s, data)
    ex = Executor(s, broker, RiskManager(s), Journal(tmp_path))
    return s, data, broker, ex


def test_executor_opens_resizes_flips_and_closes(stack):
    s, data, broker, ex = stack
    r = ex.set_target("BTC", 5000, stop_loss_pct=2)
    assert r["status"] == "executed"
    assert broker.account().positions["BTC"].notional == pytest.approx(5000)
    assert broker.get_protection("BTC")["stop_px"] == pytest.approx(98.0)
    assert ex.set_target("BTC", 5100, stop_loss_pct=4)["status"] == "protection_updated"  # inside the band
    assert broker.get_protection("BTC")["stop_px"] == pytest.approx(96.0)
    assert ex.set_target("BTC", -3000, stop_loss_pct=2)["status"] == "executed"
    pos = broker.account().positions["BTC"]
    assert pos.notional == pytest.approx(-3000)
    assert broker.get_protection("BTC")["stop_px"] == pytest.approx(102.0)
    assert ex.set_target("BTC", 0)["status"] == "executed"
    assert "BTC" not in broker.account().positions


def test_risk_limits_clamp_and_reject(stack):
    s, data, broker, ex = stack
    assert ex.set_target("DOGE", 1000, stop_loss_pct=2)["status"] == "rejected"  # not an allowed coin
    r = ex.set_target("BTC", 50_000, stop_loss_pct=50)
    assert r["approved_target_usd"] == pytest.approx(10_000)  # per-coin cap = 100 % of equity
    assert broker.get_protection("BTC")["stop_px"] == pytest.approx(90.0)  # stop clamped to 10 %
    ex.set_target("ETH", 10_000, stop_loss_pct=2)
    acct = broker.account()
    assert acct.gross_exposure <= s.max_leverage * acct.equity + 1


def test_daily_loss_limit_blocks_new_risk_but_allows_exit(stack):
    s, data, broker, ex = stack
    ex.set_target("BTC", 10_000, stop_loss_pct=10)
    data.prices["BTC"] = 94.0  # -6 % on 1x: beyond the 5 % daily limit, inside the stop
    assert ex.set_target("ETH", 1000, stop_loss_pct=2)["status"] == "rejected"
    assert ex.set_target("BTC", 0)["status"] == "executed"


def test_drawdown_halts_and_resume_clears(stack):
    s, data, broker, ex = stack
    ex.set_target("BTC", 30_000, stop_loss_pct=10)  # 3x, clamped per coin to 1x
    ex.set_target("ETH", 10_000, stop_loss_pct=10)
    data.prices.update(BTC=91.0, ETH=9.1)
    broker.state["protection"].clear()
    ex.risk.state["high_water_mark"] = 12_500  # as if equity had been higher before
    ex.account()
    assert ex.risk.state["halted"]
    assert ex.set_target("BTC", 20_000, stop_loss_pct=2)["status"] == "rejected"
    ex.flatten_all()
    assert not broker.account().positions
    ex.risk.resume(broker.account().equity)
    assert not ex.risk.state["halted"]


def test_paper_stop_triggers(stack):
    s, data, broker, ex = stack
    ex.set_target("BTC", 5000, stop_loss_pct=2)
    data.prices["BTC"] = 97.0
    acct = broker.account()
    assert "BTC" not in acct.positions
    assert acct.equity == pytest.approx(10_000 - 150)


def test_dry_run_places_nothing(stack):
    s, data, broker, ex = stack
    ex.dry_run = True
    assert ex.set_target("BTC", 5000, stop_loss_pct=2)["status"] == "dry_run"
    assert not broker.account().positions


def test_risk_check_reduction_always_allowed(tmp_path):
    s = Settings(state_dir=tmp_path, coins=["BTC"])
    rm = RiskManager(s)
    rm.halt("test")
    acct = AccountState(1000, 0, 0, {"BTC": Position("BTC", 10, 100, 100)})
    assert rm.check("BTC", 500, acct).approved
    assert not rm.check("BTC", 1500, acct).approved


# ------------------------------------------------------------------ agent loop with a scripted model
class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class ScriptedClient:
    """Stands in for the OpenAI/DeepSeek client: replays canned assistant turns."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.chat = _Obj(completions=_Obj(create=self._create))

    def _create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw["messages"])))
        turn = self.turns.pop(0)
        calls = [
            _Obj(id=f"call_{i}", function=_Obj(name=name, arguments=json.dumps(args)))
            for i, (name, args) in enumerate(turn.get("calls", []))
        ]
        msg = _Obj(content=turn.get("text", ""), tool_calls=calls or None, reasoning_content=turn.get("reasoning"))
        return _Obj(choices=[_Obj(message=msg, finish_reason="tool_calls" if calls else "stop")], usage=_Obj(prompt_tokens=10, completion_tokens=5))


def test_agent_cycle_executes_tools_and_journals(stack):
    from hyperbot.agent import TradingAgent

    s, data, broker, ex = stack
    client = ScriptedClient(
        [
            {"reasoning": "check the account first", "calls": [("get_account", {})]},
            {
                "calls": [
                    ("set_target_position", {"coin": "BTC", "target_usd": 2000, "stop_loss_pct": 3, "take_profit_pct": 0, "thesis": "test"}),
                    ("set_target_position", {"coin": "DOGE", "target_usd": 500, "stop_loss_pct": 3, "take_profit_pct": 0, "thesis": "not allowed"}),
                ]
            },
            {"text": "Opened a small BTC long."},
        ]
    )
    summary = TradingAgent(s, ex, data, client=client).run_cycle()
    assert summary["report"] == "Opened a small BTC long."
    assert summary["actions"][0]["result"]["status"] == "executed"
    assert "error" in summary["actions"][1]["result"]  # DOGE is refused before it reaches the broker
    assert broker.account().positions["BTC"].notional == pytest.approx(2000)
    # The model's reasoning and every tool result are sent back on the next request.
    second = client.requests[1]
    assert second[2]["reasoning_content"] == "check the account first"
    assert second[3]["role"] == "tool" and "equity_usd" in second[3]["content"]
    assert Journal(s.state_dir).tail(1)[0]["kind"] == "agent_cycle"


# ------------------------------------------------------------------ autopilot
def _card(rows):
    base = {"oos_trades": 20, "oos_folds": 3, "oos_folds_positive": 3, "fitted_sharpe": 1.0, "atr_pct": 1.0, "live_params": {}}
    return {"generated": "2026-01-01T00:00:00+00:00", "days": 180, "rows": [{**base, **r} for r in rows]}


def test_rule_selection_and_playbook_validation(stack):
    from hyperbot.auto import make_playbook, rule_based_allocations

    s, *_ = stack
    card = _card(
        [
            {"coin": "BTC", "interval": "4h", "strategy": "tsmom", "oos_sharpe": 1.5},
            {"coin": "BTC", "interval": "1h", "strategy": "macd", "oos_sharpe": -1.0},
            {"coin": "ETH", "interval": "4h", "strategy": "ema_cross", "oos_sharpe": 0.9, "oos_folds_positive": 1},
        ]
    )
    allocs = rule_based_allocations(s, card)
    assert [(a["coin"], a["strategy"]) for a in allocs] == [("BTC", "tsmom")]
    pb, warnings = make_playbook(
        s,
        card,
        [
            {"coin": "btc", "interval": "4h", "strategy": "tsmom", "weight": 2.5, "stop_loss_pct": 0},
            {"coin": "ETH", "interval": "4h", "strategy": "ema_cross", "weight": 5, "stop_loss_pct": 50},
            {"coin": "DOGE", "interval": "4h", "strategy": "tsmom", "weight": 1, "stop_loss_pct": 2},
            {"coin": "BTC", "interval": "1h", "strategy": "nonsense", "weight": 1, "stop_loss_pct": 2},
        ],
        risk_scale=7,
        source="llm",
    )
    assert [a["coin"] for a in pb["allocations"]] == ["BTC", "ETH"]
    assert sum(a["weight"] for a in pb["allocations"]) <= s.max_leverage + 1e-6
    assert pb["allocations"][0]["stop_loss_pct"] == 3.0  # 3 x ATR
    assert pb["allocations"][1]["stop_loss_pct"] == s.max_stop_loss_pct
    assert pb["risk_scale"] == 1.0
    assert len(warnings) >= 2


def test_portfolio_runner_follows_signals_and_blocks_after_stop(stack, monkeypatch):
    from hyperbot import auto

    s, data, broker, ex = stack
    signal = {"v": 1.0}
    monkeypatch.setattr(auto, "load_history", lambda *a, **k: make_df(n=50))
    monkeypatch.setattr(auto, "get_strategy", lambda name, **p: Fixed(signal["v"]))
    pb = {"risk_scale": 0.5, "allocations": [{"coin": "BTC", "interval": "1h", "strategy": "x", "params": {}, "weight": 0.8, "stop_loss_pct": 2.0}]}
    runner = auto.PortfolioRunner(s, ex)

    assert runner.step(pb)[0]["status"] == "executed"
    assert broker.account().positions["BTC"].notional == pytest.approx(4000)  # 0.8 x 0.5 x 10k
    stop = broker.get_protection("BTC")["stop_px"]

    data.prices["BTC"] = 99.0  # the stop must not be dragged down with the price
    assert runner.step(pb)[0]["status"] == "hold"
    assert broker.get_protection("BTC")["stop_px"] == stop

    data.prices["BTC"] = 97.0  # stop hit
    assert runner.step(pb)[0]["status"] == "noop"
    assert "BTC" not in broker.account().positions
    assert runner.step(pb)[0]["status"] == "noop"  # still long signal: stays out

    signal["v"] = -1.0  # the strategy changes side: trading resumes
    assert runner.step(pb)[0]["status"] == "executed"
    assert broker.account().positions["BTC"].size < 0

    assert runner.step({"allocations": []})[0]["status"] == "executed"  # empty playbook = flat
    assert not broker.account().positions


def test_autopilot_falls_back_to_rules_without_llm(stack, monkeypatch):
    from hyperbot import auto

    s, data, broker, ex = stack
    card = _card([{"coin": "BTC", "interval": "4h", "strategy": "tsmom", "oos_sharpe": 1.5}])
    card["generated"] = auto.utc_now().isoformat()
    monkeypatch.setattr(auto, "build_scorecard", lambda *a, **k: card)
    monkeypatch.setattr(auto, "load_history", lambda *a, **k: make_df(n=50))
    monkeypatch.setattr(auto, "get_strategy", lambda name, **p: Fixed(1.0))
    pilot = auto.AutoPilot(s, ex, use_llm=False, log=lambda m: None)
    monkeypatch.setattr(auto, "load_scorecard", lambda s_: None)
    results = pilot.tick()
    assert results[0]["status"] == "executed"
    assert auto.load_playbook(s)["source"] == "rules"
    assert broker.account().positions["BTC"].notional == pytest.approx(10_000)


def test_portfolio_agent_sets_playbook(stack, monkeypatch):
    from hyperbot import auto

    s, data, broker, ex = stack
    card = _card([{"coin": "BTC", "interval": "4h", "strategy": "tsmom", "oos_sharpe": 1.5}])
    monkeypatch.setattr(auto, "load_scorecard", lambda s_: card)
    client = ScriptedClient(
        [
            {"calls": [("get_research", {}), ("get_playbook", {})]},
            {"calls": [("set_playbook", {"allocations": [{"coin": "BTC", "interval": "4h", "strategy": "tsmom", "weight": 0.5, "stop_loss_pct": 0}], "risk_scale": 0.8, "rationale": "best oos"})]},
            {"text": "Running BTC momentum at half weight."},
        ]
    )
    agent = auto.PortfolioAgent(s, ex, data, client=client, runner=auto.PortfolioRunner(s, ex))
    names = {t["function"]["name"] for t in client_tools(agent)}
    assert "set_playbook" in names and "set_target_position" not in names
    summary = agent.run_cycle()
    assert summary["actions"][0]["result"]["status"] == "saved"
    pb = auto.load_playbook(s)
    assert pb["source"] == "llm" and pb["risk_scale"] == 0.8 and pb["allocations"][0]["weight"] == 0.5


def client_tools(agent):
    from hyperbot.agent import openai_tools

    return openai_tools(agent.tool_defs)


# ------------------------------------------------------------------ realistic simulation
def test_walk_book_and_impact():
    from hyperbot.exchange import sqrt_impact_bps, walk_book

    asks = [(100.0, 1.0), (101.0, 1.0), (102.0, 1.0)]
    assert walk_book(asks, 0.5) == pytest.approx(100.0)
    assert walk_book(asks, 2.0) == pytest.approx(100.5)
    assert walk_book(asks, 6.0) > 102.0  # bigger than the visible book: worse than the last level
    assert walk_book([], 1.0) is None
    small, big = sqrt_impact_bps(1e4, 1e9, 3.0), sqrt_impact_bps(1e6, 1e9, 3.0)
    assert big == pytest.approx(small * 10)  # 100x the size, 10x the impact
    assert sqrt_impact_bps(1e4, 0, 3.0) == 0.0


class BookData(FakeData):
    """Fake market with an order book, volume and funding."""

    def book_levels(self, coin):
        m = self.prices[coin]
        return [(m * 0.999, 10.0), (m * 0.99, 100.0)], [(m * 1.001, 10.0), (m * 1.01, 100.0)]

    def asset_contexts(self):
        return {c: {"volume_24h_usd": 1e8, "funding_hourly": 0.0001} for c in self.prices}

    def daily_vol_pct(self, coin):
        return 4.0


def test_paper_fills_pay_spread_depth_impact_and_funding(tmp_path):
    import time as _time

    s = Settings(state_dir=tmp_path, coins=["BTC"], paper_equity=1_000_000, taker_fee=0.0, max_leverage=3)
    data = BookData()
    broker = PaperBroker(s, data)
    f1 = broker.market("BTC", 5.0)  # $500: inside the first level
    assert f1.price > 100.1  # spread plus a little impact
    f2 = broker.market("BTC", 50.0)  # $5,000: eats into the second level
    assert f2.price > f1.price  # deeper book + bigger impact + pressure from the first buy
    costs = broker.state["costs"]
    assert costs["spread_and_depth"] > 0 and costs["impact"] > 0
    # Selling straight after a buy gets no rebate from our own buy pressure.
    sell = broker.market("BTC", -5.0)
    assert sell.price < 99.9
    # Funding: a long pays when the rate is positive.
    broker.state["funding_time"] = _time.time() - 2 * 3600
    before = broker.state["cash"]
    broker.account()
    assert broker.state["cash"] < before
    assert costs["funding"] == pytest.approx(50.0 * 100.0 * 0.0001 * 2, rel=0.01)


def test_sim_tracker_reports(tmp_path):
    from hyperbot.sim import SimTracker

    s = Settings(state_dir=tmp_path, coins=["BTC", "ETH"], paper_equity=5000, taker_fee=0.0, slippage_bps=0.0)
    data = FakeData()
    broker = PaperBroker(s, data)
    tracker = SimTracker(s, broker)
    broker.market("BTC", 25.0)  # $2,500 long
    tracker.record(broker.account(), data.mids())
    data.prices.update(BTC=110.0, ETH=9.0)
    tracker.record(broker.account(), data.mids())
    rep = tracker.report()
    assert rep["starting_amount"] == 5000
    assert rep["trading_return_pct"] == pytest.approx(5.0)  # +10% on half the account
    assert rep["buy_and_hold_return_pct"] == pytest.approx(0.0)  # +10% and -10%
    assert rep["trades"] == 1


# ------------------------------------------------------------------ service, API
@pytest.fixture
def service(stack, monkeypatch):
    from hyperbot import service as svc_mod

    s, data, broker, ex = stack
    monkeypatch.setattr(svc_mod, "make_broker", lambda settings: broker)
    svc = svc_mod.Service(s)
    svc.data = svc.tools.data = data
    return svc


def test_service_trades_through_risk_engine(service):
    assert service.call("account")["equity_usd"] == 10_000
    r = service.call("set_target", {"coin": "BTC", "target_usd": "50000", "stop_loss_pct": 2})
    assert r["status"] == "executed" and r["approved_target_usd"] == pytest.approx(10_000)  # clamped
    assert service.call("set_target", {"coin": "DOGE", "target_usd": 100, "stop_loss_pct": 2})["status"] == "rejected"
    assert service.call("kill")["halted"] is True
    assert service.call("account")["positions"] == []
    assert service.call("set_target", {"coin": "BTC", "target_usd": 100, "stop_loss_pct": 2})["status"] == "rejected"
    assert service.call("resume") == {"halted": False}
    assert "error" in service.call("nope")
    assert "error" in service.call("signals", {})  # missing argument comes back as an error, not a crash
    assert "ema_cross" in service.call("strategies")


def test_api_auth_and_methods(service):
    from fastapi.testclient import TestClient

    from hyperbot.api import create_app

    client = TestClient(create_app(service, token="secret"))
    assert client.get("/health").json()["ok"] is True
    assert client.get("/v1/account").status_code == 401
    h = {"Authorization": "Bearer secret"}
    assert client.get("/v1/account", headers=h).json()["equity_usd"] == 10_000
    assert "set_target" in client.get("/v1", headers=h).json()["operations"]
    assert client.get("/v1/set_target", headers=h).status_code == 405  # trading needs POST
    r = client.post("/v1/set_target", headers=h, json={"coin": "ETH", "target_usd": -1000, "stop_loss_pct": 3})
    assert r.json()["status"] == "executed"
    assert client.post("/v1/unknown", headers=h, json={}).status_code == 404


# ------------------------------------------------------------------ AI cost and net profit
def test_ai_cost_is_subtracted_from_profit(stack):
    from hyperbot.agent import TradingAgent
    from hyperbot.risk import AICostLedger
    from hyperbot.sim import SimTracker

    s, data, broker, ex = stack
    s.llm_price_input, s.llm_price_input_cached, s.llm_price_output = 1_000_000.0, 100_000.0, 2_000_000.0  # $1 / $0.10 / $2 per token
    client = ScriptedClient(
        [
            {"calls": [("set_target_position", {"coin": "BTC", "target_usd": 5000, "stop_loss_pct": 5, "take_profit_pct": 0, "thesis": "t"})]},
            {"text": "done"},
        ]
    )
    summary = TradingAgent(s, ex, data, client=client).run_cycle()
    # Two model calls, each 10 prompt + 5 completion tokens, nothing cached: 2 * (10*$1 + 5*$2) = $40
    assert summary["cost_usd"] == pytest.approx(40.0)
    assert AICostLedger(s).load()["cycles"] == 1
    assert AICostLedger(s).price(10, 4, 0) == pytest.approx(6 * 1.0 + 4 * 0.1)  # cached tokens are cheaper

    data.prices["BTC"] = 102.0  # +2% on $5,000 = +$100 trading profit
    acct = ex.account()
    pnl = ex.risk.pnl(acct.equity)
    assert pnl["trading_pnl_usd"] == pytest.approx(100.0)
    assert pnl["ai_cost_usd"] == pytest.approx(40.0)
    assert pnl["net_pnl_usd"] == pytest.approx(60.0)

    tracker = SimTracker(s, broker)
    tracker.record(acct, data.mids())
    rep = tracker.report()
    assert rep["trading_pnl_usd"] == pytest.approx(100.0) and rep["net_pnl_usd"] == pytest.approx(60.0)
    assert rep["net_return_pct"] == pytest.approx(0.6)

    data.prices["BTC"] = 99.0  # a trading loss gets bigger once the AI bill is added
    assert ex.risk.pnl(ex.account().equity)["net_pnl_usd"] == pytest.approx(-50.0 - 40.0)


# ------------------------------------------------------------------ luck-adjusted statistics
def test_sharpe_statistics_separate_edge_from_luck():
    from hyperbot.stats import deflated_sharpe, expected_max_sharpe, half_kelly, probabilistic_sharpe

    rng = np.random.default_rng(0)
    edge = pd.Series(rng.normal(0.002, 0.01, 2000))  # real edge, long record
    noise = pd.Series(rng.normal(0.0, 0.01, 2000))
    short = edge.iloc[:60]
    assert probabilistic_sharpe(edge) > 0.99
    assert 0.05 < probabilistic_sharpe(noise) < 0.95
    assert probabilistic_sharpe(short) < probabilistic_sharpe(edge)  # same edge, less evidence
    # The more candidates you tried, the higher the bar and the lower the deflated figure.
    assert expected_max_sharpe(100, 0.001) > expected_max_sharpe(10, 0.001) > 0
    assert deflated_sharpe(short, 200, 0.02) < deflated_sharpe(short, 2, 0.02) <= probabilistic_sharpe(short)
    assert half_kelly(edge) == pytest.approx(0.5 * edge.mean() / edge.var(), rel=1e-9)
    assert half_kelly(-edge) < 0


def test_rule_weights_use_risk_parity_confidence_and_kelly(stack):
    from hyperbot.auto import finalize_rows, rule_based_allocations
    from hyperbot.stats import sharpe_stats

    s, *_ = stack
    rng = np.random.default_rng(1)
    base = {"oos_trades": 20, "oos_folds": 3, "oos_folds_positive": 3, "fitted_sharpe": 1.0, "atr_pct": 1.0, "live_params": {}, "interval": "4h"}
    calm = pd.Series(rng.normal(0.003, 0.01, 800))
    wild = pd.Series(rng.normal(0.006, 0.04, 800))
    dud = pd.Series(rng.normal(-0.0005, 0.01, 800))  # slightly losing: must be rejected
    rows = finalize_rows(
        [
            {**base, "coin": "BTC", "strategy": "tsmom", "oos_sharpe": 3.0, "oos_volatility_pct": 20.0, "_stats": sharpe_stats(calm)},
            {**base, "coin": "ETH", "strategy": "ema_cross", "oos_sharpe": 2.0, "oos_volatility_pct": 80.0, "_stats": sharpe_stats(wild)},
            {**base, "coin": "ETH", "strategy": "macd", "oos_sharpe": 0.6, "oos_volatility_pct": 20.0, "_stats": sharpe_stats(dud)},
        ]
    )
    assert all("_stats" not in r and 0 <= r["dsr"] <= r["psr"] <= 1 for r in rows)
    allocs = {a["strategy"]: a["weight"] for a in rule_based_allocations(s, {"rows": rows})}
    assert "macd" not in allocs  # no evidence it beats zero
    assert allocs["tsmom"] == pytest.approx(4 * allocs["ema_cross"], rel=0.01)  # a quarter of the volatility, four times the weight
    assert sum(allocs.values()) <= 1.0


def test_manager_can_take_profit_close_and_reenter(stack, monkeypatch):
    from hyperbot import auto

    s, data, broker, ex = stack
    monkeypatch.setattr(auto, "load_history", lambda *a, **k: make_df(n=50))
    monkeypatch.setattr(auto, "get_strategy", lambda name, **p: Fixed(1.0))
    monkeypatch.setattr(auto, "tradable_strategies", lambda: ["tsmom"])
    monkeypatch.setattr(auto, "load_scorecard", lambda s_: None)
    runner = auto.PortfolioRunner(s, ex)
    agent = auto.PortfolioAgent(s, ex, data, client=object(), runner=runner)
    alloc = {"coin": "BTC", "interval": "4h", "strategy": "tsmom", "weight": 0.5, "stop_loss_pct": 5, "take_profit_pct": 4}

    agent.tool_set_playbook({"allocations": [alloc], "risk_scale": 1.0, "rationale": "r"})
    runner.step(auto.load_playbook(s))
    assert broker.get_protection("BTC") == {"stop_px": pytest.approx(95.0), "tp_px": pytest.approx(104.0)}

    data.prices["BTC"] = 105.0  # take-profit hit: banked, and not re-bought on the next tick
    runner.step(auto.load_playbook(s))
    assert "BTC" not in broker.account().positions
    assert broker.account().equity == pytest.approx(10_250)
    assert runner.step(auto.load_playbook(s))[0]["status"] == "noop"

    agent.tool_set_playbook({"allocations": [alloc], "risk_scale": 1.0, "rationale": "re-enter"})  # a new decision re-opens
    assert runner.step(auto.load_playbook(s))[0]["status"] == "executed"

    out = agent.tool_close_position({"coin": "BTC", "reason": "bank it"})  # manual close holds the coin closed
    assert out["status"] == "executed" and "held_closed" in out
    assert runner.step(auto.load_playbook(s))[0]["status"] == "noop"
    assert not broker.account().positions
