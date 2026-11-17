"""Current quantitative read on a coin: indicators plus what every strategy says right now."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .backtest import BacktestConfig, run_backtest
from .data import infer_interval_ms
from .strategies import YEAR_MS, get_strategy, tradable_strategies


def _r(x, d: int = 2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, d) if np.isfinite(x) else None


def quant_snapshot(df: pd.DataFrame) -> dict:
    """Indicators and strategy signals as of the last closed bar in `df`."""
    c = df["close"]
    step = infer_interval_ms(df)
    bars_per_year = YEAR_MS / step
    bars_per_day = max(int(86_400_000 / step), 1)
    price = float(c.iloc[-1])

    def ret(bars: int):
        return _r((price / c.iloc[-1 - bars] - 1) * 100) if len(c) > bars else None

    lo, mid, hi = ind.bollinger(c, 20, 2.0)
    _, _, hist = ind.macd(c)
    ema20, ema50, ema200 = ind.ema(c, 20).iloc[-1], ind.ema(c, 50).iloc[-1], ind.ema(c, 200).iloc[-1]
    adx = ind.adx(df).iloc[-1]
    indicators = {
        "price": price,
        "return_1d_pct": ret(bars_per_day),
        "return_7d_pct": ret(bars_per_day * 7),
        "return_30d_pct": ret(bars_per_day * 30),
        "rsi_14": _r(ind.rsi(c).iloc[-1], 1),
        "adx_14": _r(adx, 1),
        "regime": "trending" if adx >= 25 else "ranging" if adx <= 18 else "mixed",
        "atr_pct": _r(ind.atr(df).iloc[-1] / price * 100),
        "realized_vol_annual_pct": _r(ind.realized_vol(c, min(72, len(c) - 1), bars_per_year).iloc[-1] * 100, 1),
        "zscore_20": _r(ind.zscore(c).iloc[-1]),
        "bollinger_position": _r((price - lo.iloc[-1]) / (hi.iloc[-1] - lo.iloc[-1])) if hi.iloc[-1] != lo.iloc[-1] else None,
        "macd_histogram_sign": int(np.sign(hist.iloc[-1])) if np.isfinite(hist.iloc[-1]) else 0,
        "price_vs_ema20_pct": _r((price / ema20 - 1) * 100),
        "price_vs_ema50_pct": _r((price / ema50 - 1) * 100),
        "price_vs_ema200_pct": _r((price / ema200 - 1) * 100),
        "ema_stack": "bullish" if ema20 > ema50 > ema200 else "bearish" if ema20 < ema50 < ema200 else "mixed",
    }
    if "funding" in df.columns and df["funding"].abs().sum():
        indicators["funding_24h_avg_annual_pct"] = _r(df["funding"].iloc[-bars_per_day:].mean() * bars_per_year * 100)

    signals = {}
    for name in tradable_strategies():
        try:
            signals[name] = _r(get_strategy(name).signal(df).iloc[-1])
        except Exception as e:  # one broken strategy should not hide the rest
            signals[name] = f"error: {e}"
    numeric = [v for v in signals.values() if isinstance(v, (int, float))]
    return {
        "as_of_bar_close": str(df.index[-1] + pd.Timedelta(milliseconds=step)),
        "bars": len(df),
        "indicators": indicators,
        "strategy_signals": signals,
        "consensus": _r(sum(numeric) / len(numeric)) if numeric else None,
        "legend": "signals are target exposure: +1 long, -1 short, 0 flat; consensus is their average",
    }


def rank_strategies(df: pd.DataFrame, cfg: BacktestConfig | None = None, names: list[str] | None = None) -> pd.DataFrame:
    """Backtest every strategy with default parameters on the same data."""
    rows = []
    for name in names or ["buy_hold", *tradable_strategies()]:
        m = run_backtest(df, get_strategy(name), cfg).metrics
        rows.append(
            {
                "strategy": name,
                "return_pct": m["total_return_pct"],
                "sharpe": m["sharpe"],
                "max_dd_pct": m["max_drawdown_pct"],
                "trades": m["trades"],
                "win_rate_pct": m["win_rate_pct"],
                "profit_factor": m["profit_factor"],
                "in_market_pct": m["time_in_market_pct"],
                "fees": m.get("fees_paid", 0.0),
                "funding": m.get("funding_paid", 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
