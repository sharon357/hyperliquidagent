"""Bar-by-bar backtester for perpetual futures.

What it models: next-bar-open execution (no look-ahead), taker fees, slippage,
hourly funding payments, leverage, stop-loss / take-profit hit inside a bar,
and liquidation. What it does not model: order-book depth, partial fills, latency.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import infer_interval_ms
from .strategies import YEAR_MS, Strategy, get_strategy


@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    leverage: float = 1.0  # notional = signal * leverage * equity
    taker_fee: float = 0.00045
    slippage_bps: float = 2.0
    stop_loss_pct: float | None = None  # % from the average entry price
    take_profit_pct: float | None = None
    trail_pct: float | None = None  # trailing stop: this far behind the best price since entry
    breakeven_at_pct: float | None = None  # once this far in profit, the stop is at least the entry
    maintenance_margin: float = 0.02  # liquidated when equity < this share of notional
    rebalance_threshold: float = 0.1  # smallest change in target exposure worth trading
    allow_short: bool = True


@dataclass
class BacktestResult:
    strategy: str
    params: dict
    config: BacktestConfig
    equity: pd.Series
    exposure: pd.Series  # signed notional / equity at each bar close
    trades: pd.DataFrame
    metrics: dict


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    cfg: BacktestConfig | None = None,
    trade_from: pd.Timestamp | None = None,
) -> BacktestResult:
    """Simulate `strategy` on `df`. With `trade_from`, earlier bars are warm-up only."""
    cfg = cfg or BacktestConfig()
    sig = strategy.signal(df).reindex(df.index).fillna(0.0).clip(-1, 1).to_numpy(dtype=float)
    if not cfg.allow_short:
        sig = np.clip(sig, 0, 1)
    # Decision at the close of bar t-1 is executed at the open of bar t.
    target = np.roll(sig, 1) * cfg.leverage
    target[0] = 0.0

    start = 0
    if trade_from is not None:
        start = int(df.index.searchsorted(trade_from))
        target[:start] = 0.0

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    f = df["funding"].to_numpy(dtype=float) if "funding" in df.columns else np.zeros(len(df))
    times = df.index

    n = len(df)
    eq = np.full(n, cfg.initial_equity)
    expo = np.zeros(n)
    slip = cfg.slippage_bps / 1e4
    sl = cfg.stop_loss_pct / 100 if cfg.stop_loss_pct else None
    tp = cfg.take_profit_pct / 100 if cfg.take_profit_pct else None
    trail = cfg.trail_pct / 100 if cfg.trail_pct else None
    breakeven = cfg.breakeven_at_pct / 100 if cfg.breakeven_at_pct else None
    best_px = 0.0  # best price since entry, as of the previous bar (so a bar's high cannot rescue its own low)
    mm = cfg.maintenance_margin

    equity = cfg.initial_equity
    units = 0.0
    entry_px = 0.0
    last_target = 0.0
    blocked = 0.0  # after a stop/liquidation, stay flat until the signal changes side
    fees = slippage_cost = funding_paid = 0.0
    liquidations = 0
    trades: list[dict] = []
    open_trade: dict | None = None

    def trade_to(desired: float, price: float, i: int, reason: str) -> None:
        nonlocal equity, units, entry_px, fees, slippage_cost, open_trade, best_px
        delta = desired - units
        if delta == 0:
            return
        notional = abs(delta) * price
        equity -= notional * (cfg.taker_fee + slip)
        fees += notional * cfg.taker_fee
        slippage_cost += notional * slip
        if units == 0:  # opening
            entry_px = price
            best_px = price
            open_trade = {
                "entry_time": times[i],
                "side": "long" if desired > 0 else "short",
                "entry_px": price,
                "equity_before": equity + notional * (cfg.taker_fee + slip),
            }
        elif abs(desired) > abs(units):  # adding
            entry_px = (entry_px * abs(units) + price * abs(delta)) / abs(desired)
        units = desired
        if units == 0 and open_trade is not None:  # closed
            pnl = equity - open_trade["equity_before"]
            trades.append(
                {
                    **open_trade,
                    "exit_time": times[i],
                    "exit_px": price,
                    "pnl": pnl,
                    "return_pct": pnl / open_trade["equity_before"] * 100,
                    "reason": reason,
                }
            )
            open_trade = None

    def protect(lo: float, hi: float, ref: float, i: int) -> None:
        """Apply stop-loss, liquidation and take-profit for the price range [lo, hi]."""
        nonlocal equity, last_target, blocked, liquidations
        if units == 0:
            return
        long = units > 0
        denom = units * (1 - mm) if long else units * (1 + mm)
        liq_px = (units * ref - equity) / denom
        hit_px = None
        reason = ""
        if long:
            stop_px = entry_px * (1 - sl) if sl else -np.inf
            if trail:
                stop_px = max(stop_px, best_px * (1 - trail))
            if breakeven and best_px >= entry_px * (1 + breakeven):
                stop_px = max(stop_px, entry_px * 1.001)
            trigger = max(stop_px, liq_px)
            if lo <= trigger:
                hit_px = min(trigger, hi)
                reason = "liquidation" if liq_px >= stop_px else ("trailing_stop" if stop_px > entry_px * (1 - (sl or 1)) else "stop_loss")
            elif tp and hi >= entry_px * (1 + tp):
                hit_px, reason = max(entry_px * (1 + tp), lo), "take_profit"
        else:
            stop_px = entry_px * (1 + sl) if sl else np.inf
            if trail:
                stop_px = min(stop_px, best_px * (1 + trail))
            if breakeven and best_px <= entry_px * (1 - breakeven):
                stop_px = min(stop_px, entry_px * 0.999)
            trigger = min(stop_px, liq_px) if liq_px > 0 else stop_px
            if hi >= trigger:
                hit_px = max(trigger, lo)
                reason = "liquidation" if 0 < liq_px <= stop_px else ("trailing_stop" if stop_px < entry_px * (1 + (sl or 1e9)) else "stop_loss")
            elif tp and lo <= entry_px * (1 - tp):
                hit_px, reason = min(entry_px * (1 - tp), hi), "take_profit"
        if hit_px is None:
            return
        side = np.sign(units)
        equity += units * (hit_px - ref)
        trade_to(0.0, hit_px, i, reason)
        last_target = 0.0
        if reason != "take_profit":
            blocked = side  # as live: stay out until the signal changes side
        if reason == "liquidation":
            liquidations += 1

    prev_close = o[start] if n else 0.0
    for i in range(start, n):
        if units:
            equity += units * (o[i] - prev_close)  # overnight / gap move
        protect(o[i], o[i], o[i], i)  # a gap through the stop fills at the open

        t = target[i]
        if blocked:
            if np.sign(t) == blocked:
                t = 0.0
            else:
                blocked = 0.0
        if equity > 0:
            flip = units != 0 and t != 0 and np.sign(t) != np.sign(units)
            if flip or (t == 0 and units != 0) or abs(t - last_target) >= cfg.rebalance_threshold:
                if flip:
                    trade_to(0.0, o[i], i, "flip")
                reason = "signal"
                trade_to(t * equity / o[i], o[i], i, reason)
                last_target = t

        protect(low[i], h[i], o[i], i)
        if units:
            best_px = max(best_px, h[i]) if units > 0 else min(best_px, low[i])
            equity += units * (c[i] - o[i])
            pay = units * c[i] * f[i]  # longs pay when funding is positive
            equity -= pay
            funding_paid += pay
        prev_close = c[i]
        if equity <= 0:
            equity = 0.0
            if units:
                trade_to(0.0, c[i], i, "bust")
            eq[i:] = 0.0
            break
        eq[i] = equity
        expo[i] = units * c[i] / equity

    if open_trade is not None:  # mark the still-open trade at the last close
        pnl = equity - open_trade["equity_before"]
        trades.append(
            {
                **open_trade,
                "exit_time": times[-1],
                "exit_px": c[-1],
                "pnl": pnl,
                "return_pct": pnl / open_trade["equity_before"] * 100,
                "reason": "open",
            }
        )

    equity_s = pd.Series(eq, index=df.index, name="equity").iloc[start:]
    expo_s = pd.Series(expo, index=df.index, name="exposure").iloc[start:]
    trades_df = pd.DataFrame(trades).drop(columns=["equity_before"], errors="ignore")
    metrics = compute_metrics(equity_s, expo_s, trades_df, df["close"].iloc[start:], cfg.initial_equity)
    metrics.update(
        fees_paid=round(fees, 2),
        slippage_cost=round(slippage_cost, 2),
        funding_paid=round(funding_paid, 2),
        liquidations=liquidations,
    )
    return BacktestResult(strategy.name, dict(strategy.params), cfg, equity_s, expo_s, trades_df, metrics)


def compute_metrics(
    equity: pd.Series, exposure: pd.Series, trades: pd.DataFrame, close: pd.Series, initial: float
) -> dict:
    if len(equity) < 2:
        return {"bars": len(equity), "total_return_pct": 0.0, "sharpe": 0.0, "trades": 0}
    bars_per_year = YEAR_MS / infer_interval_ms(equity.to_frame())
    years = len(equity) / bars_per_year
    rets = equity.pct_change().fillna(equity.iloc[0] / initial - 1).replace([np.inf, -np.inf], 0.0)
    total = equity.iloc[-1] / initial - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    vol = rets.std(ddof=0) * np.sqrt(bars_per_year)
    downside = rets[rets < 0].std(ddof=0) * np.sqrt(bars_per_year)
    mean_ann = rets.mean() * bars_per_year
    peak = equity.cummax()
    dd = (equity / peak - 1).min()
    n_trades = len(trades)
    wins = trades[trades["pnl"] > 0] if n_trades else trades
    losses = trades[trades["pnl"] <= 0] if n_trades else trades
    gross_win = wins["pnl"].sum() if n_trades else 0.0
    gross_loss = -losses["pnl"].sum() if n_trades else 0.0

    def r(x: float, d: int = 2) -> float:
        return round(float(x), d) if np.isfinite(x) else 0.0

    return {
        "bars": len(equity),
        "years": r(years, 3),
        "total_return_pct": r(total * 100),
        "buy_hold_return_pct": r((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "cagr_pct": r(cagr * 100),
        "volatility_pct": r(vol * 100),
        "sharpe": r(mean_ann / vol) if vol > 0 else 0.0,
        "sortino": r(mean_ann / downside) if downside and downside > 0 else 0.0,
        "max_drawdown_pct": r(dd * 100),
        "calmar": r(cagr / abs(dd)) if dd < 0 else 0.0,
        "trades": n_trades,
        "win_rate_pct": r(len(wins) / n_trades * 100) if n_trades else 0.0,
        "profit_factor": r(gross_win / gross_loss) if gross_loss > 0 else 0.0,
        "avg_trade_pct": r(trades["return_pct"].mean()) if n_trades else 0.0,
        "time_in_market_pct": r((exposure != 0).mean() * 100),
        "final_equity": r(equity.iloc[-1]),
    }


def expand_grid(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def grid_search(
    df: pd.DataFrame,
    strategy_name: str,
    cfg: BacktestConfig | None = None,
    grid: dict | None = None,
    metric: str = "sharpe",
    min_trades: int = 5,
    trade_from: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Backtest every parameter combination. Sorted best-first by `metric`."""
    base = get_strategy(strategy_name)
    rows = []
    for params in expand_grid(grid if grid is not None else base.grid):
        res = run_backtest(df, get_strategy(strategy_name, **params), cfg, trade_from)
        rows.append({**params, **res.metrics})
    out = pd.DataFrame(rows)
    out["eligible"] = out["trades"] >= min_trades
    return out.sort_values(["eligible", metric], ascending=[False, False]).reset_index(drop=True)


def walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    cfg: BacktestConfig | None = None,
    grid: dict | None = None,
    n_splits: int = 4,
    metric: str = "sharpe",
    min_trades: int = 3,
) -> dict:
    """Expanding-window walk-forward test.

    For each fold the parameters are chosen on past data only, then traded on the
    next unseen slice. The stitched out-of-sample result is the honest estimate.
    """
    cfg = cfg or BacktestConfig()
    base = get_strategy(strategy_name)
    grid = grid if grid is not None else base.grid
    n = len(df)
    seg = n // (n_splits + 1)
    if seg < 50:
        raise ValueError("Not enough data for walk-forward: need at least 50 bars per segment")
    folds = []
    oos_returns = []
    oos_trades = []
    for k in range(1, n_splits + 1):
        train = df.iloc[: k * seg]
        test_end = (k + 1) * seg if k < n_splits else n
        test_start_time = df.index[k * seg]
        ranked = grid_search(train, strategy_name, cfg, grid, metric, min_trades)
        best = ranked.iloc[0]
        params = {key: type(grid[key][0])(best[key]) for key in grid}
        # Run on all data up to the end of the test slice so indicators are warm,
        # but only trade inside the test slice.
        res = run_backtest(df.iloc[:test_end], get_strategy(strategy_name, **params), cfg, trade_from=test_start_time)
        folds.append(
            {
                "fold": k,
                "test_start": str(test_start_time.date()),
                "test_end": str(df.index[test_end - 1].date()),
                "params": params,
                f"in_sample_{metric}": float(best[metric]),
                f"out_of_sample_{metric}": res.metrics.get(metric, 0.0),
                "out_of_sample_return_pct": res.metrics["total_return_pct"],
                "trades": res.metrics["trades"],
            }
        )
        oos_returns.append(res.equity.pct_change().fillna(res.equity.iloc[0] / cfg.initial_equity - 1))
        if not res.trades.empty:
            oos_trades.append(res.trades)
    rets = pd.concat(oos_returns)
    equity = cfg.initial_equity * (1 + rets).cumprod()
    trades = pd.concat(oos_trades) if oos_trades else pd.DataFrame(columns=["pnl", "return_pct"])
    exposure = pd.Series(1.0, index=equity.index)
    metrics = compute_metrics(equity, exposure, trades, df["close"].reindex(equity.index), cfg.initial_equity)
    metrics.pop("time_in_market_pct", None)
    is_avg = float(np.mean([f[f"in_sample_{metric}"] for f in folds]))
    oos_avg = float(np.mean([f[f"out_of_sample_{metric}"] for f in folds]))
    return {
        "strategy": strategy_name,
        "metric": metric,
        "folds": folds,
        "in_sample_avg": round(is_avg, 2),
        "out_of_sample_avg": round(oos_avg, 2),
        "oos_metrics": metrics,
        "oos_equity": equity,
        "config": asdict(cfg),
    }

