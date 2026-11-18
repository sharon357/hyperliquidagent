"""Simulation: track a live paper run, and replay the whole autopilot on history.

- SimTracker records the equity of a paper account tick by tick, next to a
  buy-and-hold benchmark, and reports the result with a cost breakdown.
- replay() answers "what would the automation have done over the last N days?":
  at each re-pick date it researches using only earlier data, picks strategies
  with the fixed rule, trades them until the next re-pick, and repeats. The LLM
  is not part of a replay, because a language model may already know what
  happened in the period being replayed.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .auto import _default_stop, _read, _write, finalize_rows, rule_based_allocations, score_frame
from .backtest import BacktestConfig, compute_metrics, run_backtest
from .config import Settings
from .data import interval_ms, load_history
from .exchange import AccountState, PaperBroker, sqrt_impact_bps
from .risk import AICostLedger, utc_now
from .strategies import get_strategy


class SimTracker:
    def __init__(self, settings: Settings, broker: PaperBroker):
        self.s = settings
        self.broker = broker
        self.csv_path = Path(settings.state_dir) / "equity.csv"
        self.meta_path = Path(settings.state_dir) / "sim_meta.json"

    def record(self, acct: AccountState, mids: dict[str, float]) -> dict:
        meta = _read(self.meta_path)
        if meta is None:
            meta = {
                "started": utc_now().isoformat(timespec="seconds"),
                "amount": self.broker.state.get("start_equity", self.s.paper_equity),
                "start_prices": {c: mids[c] for c in self.s.coins if c in mids},
            }
            _write(self.meta_path, meta)
        rel = [mids[c] / p for c, p in meta["start_prices"].items() if c in mids and p]
        benchmark = meta["amount"] * (sum(rel) / len(rel)) if rel else meta["amount"]
        row = {
            "time": utc_now().isoformat(timespec="seconds"),
            "equity": round(acct.equity, 2),
            "benchmark": round(benchmark, 2),
            "gross_exposure": round(acct.gross_exposure, 2),
        }
        new = not self.csv_path.is_file()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        return row

    def report(self) -> dict | None:
        meta = _read(self.meta_path)
        if meta is None or not self.csv_path.is_file():
            return None
        df = pd.read_csv(self.csv_path)
        if df.empty:
            return None
        amount = meta["amount"]
        eq = pd.concat([pd.Series([amount]), df["equity"]], ignore_index=True)
        days = (pd.Timestamp(df["time"].iloc[-1]) - pd.Timestamp(meta["started"])).total_seconds() / 86400
        costs = self.broker.state["costs"]
        total_cost = sum(costs.values())
        gross = float(df["equity"].iloc[-1]) - amount
        ai = AICostLedger(self.s).load()
        net = gross - ai["cost_usd"]
        return {
            "started": meta["started"],
            "days_running": round(days, 2),
            "starting_amount": amount,
            "equity_now": float(df["equity"].iloc[-1]),
            "trading_pnl_usd": round(gross, 2),
            "ai_cost_usd": round(ai["cost_usd"], 4),
            "ai_decision_cycles": ai["cycles"],
            "net_pnl_usd": round(net, 2),
            "net_return_pct": round(net / amount * 100, 2),
            "trading_return_pct": round(gross / amount * 100, 2),
            "buy_and_hold_return_pct": round((df["benchmark"].iloc[-1] / amount - 1) * 100, 2),
            "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min() * 100), 2),
            "trades": self.broker.state["trades"],
            "traded_volume_usd": round(self.broker.state["volume_usd"], 2),
            "cost_fees_usd": round(costs["fees"], 2),
            "cost_spread_and_depth_usd": round(costs["spread_and_depth"], 2),
            "cost_market_impact_usd": round(costs["impact"], 2),
            "cost_funding_usd": round(costs["funding"], 2),
            "total_costs_usd": round(total_cost, 2),
            "total_costs_pct_of_start": round(total_cost / amount * 100, 3),
        }


def _liquidity(df: pd.DataFrame) -> tuple[float, float]:
    """(average daily traded USD, daily volatility in percent) from candles."""
    usd = (df["volume"] * df["close"]).resample("1D").sum()
    daily = df["close"].resample("1D").last().pct_change().dropna()
    return float(usd.mean()), float(daily.std() * 100) if len(daily) > 5 else 3.0


def replay(
    settings: Settings,
    amount: float = 10_000.0,
    days: float = 150.0,
    interval: str = "4h",
    repick_days: int = 14,
    lookback_days: float = 150.0,
    log: Callable[[str], None] = print,
) -> dict:
    """Walk the rule-based autopilot through history, re-picking strategies as it goes."""
    step = pd.Timedelta(milliseconds=interval_ms(interval))
    frames = {c: load_history(c, interval, days + lookback_days + 3, settings.data_dir) for c in settings.coins}
    end = min(f.index[-1] for f in frames.values()) + step
    start = end - pd.Timedelta(days=days)
    earliest = max(f.index[0] for f in frames.values())
    if start - pd.Timedelta(days=lookback_days) < earliest - pd.Timedelta(days=2):
        lookback_days = (start - earliest).total_seconds() / 86400
        if lookback_days < 60:
            raise ValueError(f"Not enough {interval} history for a {days:g}-day replay. Use fewer days or a larger interval.")
        log(f"only {lookback_days:.0f} days of history before the replay window; using that as the research window")
    lookback = pd.Timedelta(days=lookback_days)
    unit = 10_000.0  # each strategy sleeve is simulated on this notional, then scaled by its weight
    base = BacktestConfig(initial_equity=unit, taker_fee=settings.taker_fee, slippage_bps=settings.slippage_bps)

    bounds = list(pd.date_range(start, end, freq=f"{repick_days}D"))
    if bounds[-1] < end:
        bounds.append(end)
    equity = amount
    curves, periods = [], []
    impact_cost = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = []
        for coin, f in frames.items():
            hist = f[(f.index >= a - lookback) & (f.index < a)]
            if len(hist) >= 250:
                rows.extend(score_frame(hist, coin, interval, base, log=lambda m: None))
        finalize_rows(rows)
        index = {(r["coin"], r["strategy"]): r for r in rows}
        allocs = rule_based_allocations(settings, {"rows": rows})
        ref = next(iter(frames.values()))
        idx = ref.index[(ref.index >= a) & (ref.index < b)]
        port = pd.Series(0.0, index=idx)
        for al in allocs:
            row = index[(al["coin"], al["strategy"])]
            f = frames[al["coin"]]
            hist = f[(f.index >= a - lookback) & (f.index < a)]
            adv, dvol = _liquidity(hist)
            sleeve = equity * al["weight"]
            imp = sqrt_impact_bps(sleeve, adv, dvol) * settings.sim_impact
            cfg = replace(base, slippage_bps=base.slippage_bps + imp, stop_loss_pct=_default_stop(settings, row["atr_pct"]))
            res = run_backtest(f[f.index < b], get_strategy(al["strategy"], **row["live_params"]), cfg, trade_from=a)
            r = res.equity.pct_change()
            r.iloc[0] = res.equity.iloc[0] / unit - 1.0
            port = port.add(al["weight"] * r.reindex(idx).fillna(0.0), fill_value=0.0)
            # slippage_cost / slippage rate = notional traded on the unit sleeve
            traded = (res.metrics["slippage_cost"] / (cfg.slippage_bps / 1e4)) if cfg.slippage_bps else 0.0
            impact_cost += traded / unit * sleeve * imp / 1e4
        curve = equity * (1 + port).cumprod()
        periods.append(
            {
                "from": str(a.date()),
                "to": str((b - step).date()),
                "picked": ", ".join(f"{al['coin']} {al['strategy']}" for al in allocs) or "(nothing passed: flat)",
                "return_pct": round((curve.iloc[-1] / equity - 1) * 100, 2) if len(curve) else 0.0,
                "equity": round(float(curve.iloc[-1]), 2) if len(curve) else round(equity, 2),
            }
        )
        log(f"{periods[-1]['from']} to {periods[-1]['to']}: {periods[-1]['return_pct']:+.2f}%  [{periods[-1]['picked']}]")
        if len(curve):
            curves.append(curve)
            equity = float(curve.iloc[-1])

    eq = pd.concat(curves)
    rel = [f["close"].reindex(eq.index).ffill() / f["close"].reindex(eq.index).ffill().iloc[0] for f in frames.values()]
    bench = amount * sum(rel) / len(rel)
    metrics = compute_metrics(eq, pd.Series(1.0, index=eq.index), pd.DataFrame(), bench, amount)
    for k in ("trades", "win_rate_pct", "profit_factor", "avg_trade_pct", "time_in_market_pct"):
        metrics.pop(k, None)
    metrics["estimated_market_impact_usd"] = round(impact_cost, 2)
    metrics["periods_profitable"] = f"{sum(p['return_pct'] > 0 for p in periods)}/{len(periods)}"
    return {"metrics": metrics, "periods": periods, "equity": eq, "benchmark": bench, "interval": interval}
