"""Trading strategies.

Every strategy maps a candle DataFrame to a *target exposure* series in [-1, 1]:
+1 fully long, -1 fully short, 0 flat. The value at bar t may only use data up to
the close of bar t; the backtester and the live runner act on it at the next bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .data import infer_interval_ms

YEAR_MS = 365 * 86_400_000


class Strategy:
    name = ""
    description = ""
    defaults: dict = {}
    grid: dict = {}  # parameter grid used by `optimize` and `walkforward`

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}. Valid: {sorted(self.defaults)}")
        self.params = {**self.defaults, **params}

    def signal(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"


REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


def _stateful(long_entry, long_exit, short_entry, short_exit, allow_short: bool = True) -> np.ndarray:
    """Hold a position from an entry condition until its exit condition."""
    n = len(long_entry)
    out = np.zeros(n)
    pos = 0.0
    for i in range(n):
        if pos > 0 and long_exit[i]:
            pos = 0.0
        elif pos < 0 and short_exit[i]:
            pos = 0.0
        if pos == 0:
            if long_entry[i]:
                pos = 1.0
            elif allow_short and short_entry[i]:
                pos = -1.0
        out[i] = pos
    return out


def _b(s: pd.Series) -> np.ndarray:
    return s.fillna(False).to_numpy(dtype=bool)


@register
class BuyHold(Strategy):
    name = "buy_hold"
    description = "Always long. The benchmark every other strategy has to beat."

    def signal(self, df):
        return pd.Series(1.0, index=df.index)


@register
class EmaCross(Strategy):
    name = "ema_cross"
    description = "Trend following: long when the fast EMA is above the slow EMA, short when below."
    defaults = {"fast": 20, "slow": 100, "allow_short": True}
    grid = {"fast": [10, 20, 50], "slow": [50, 100, 200]}

    def signal(self, df):
        p = self.params
        if p["fast"] >= p["slow"]:
            return pd.Series(0.0, index=df.index)
        diff = ind.ema(df["close"], p["fast"]) - ind.ema(df["close"], p["slow"])
        sig = np.sign(diff).fillna(0.0)
        return sig if p["allow_short"] else sig.clip(lower=0)


@register
class Macd(Strategy):
    name = "macd"
    description = "Momentum: long when the MACD histogram is positive, short when negative."
    defaults = {"fast": 12, "slow": 26, "signal": 9, "allow_short": True}
    grid = {"fast": [8, 12, 24], "slow": [26, 52, 104], "signal": [9, 18]}

    def signal(self, df):
        p = self.params
        if p["fast"] >= p["slow"]:
            return pd.Series(0.0, index=df.index)
        _, _, hist = ind.macd(df["close"], p["fast"], p["slow"], p["signal"])
        sig = np.sign(hist).fillna(0.0)
        return sig if p["allow_short"] else sig.clip(lower=0)


@register
class RsiReversion(Strategy):
    name = "rsi_reversion"
    description = "Mean reversion: buy oversold RSI, sell overbought RSI, exit at the midline."
    defaults = {"period": 14, "lower": 30, "upper": 70, "exit": 50, "allow_short": True}
    grid = {"period": [7, 14, 21], "lower": [20, 25, 30], "upper": [70, 75, 80]}

    def signal(self, df):
        p = self.params
        r = ind.rsi(df["close"], p["period"])
        pos = _stateful(_b(r < p["lower"]), _b(r > p["exit"]), _b(r > p["upper"]), _b(r < p["exit"]), p["allow_short"])
        return pd.Series(pos, index=df.index)


@register
class BollingerReversion(Strategy):
    name = "bollinger_reversion"
    description = "Mean reversion: fade closes outside the Bollinger bands, exit at the middle band."
    defaults = {"period": 20, "k": 2.0, "allow_short": True}
    grid = {"period": [20, 50, 100], "k": [1.5, 2.0, 2.5, 3.0]}

    def signal(self, df):
        p = self.params
        lo, mid, hi = ind.bollinger(df["close"], p["period"], p["k"])
        c = df["close"]
        pos = _stateful(_b(c < lo), _b(c >= mid), _b(c > hi), _b(c <= mid), p["allow_short"])
        return pd.Series(pos, index=df.index)


@register
class DonchianBreakout(Strategy):
    name = "donchian_breakout"
    description = "Breakout (turtle style): enter on a new N-bar high/low, exit on a shorter opposite channel."
    defaults = {"entry": 55, "exit": 20, "allow_short": True}
    grid = {"entry": [20, 55, 100], "exit": [10, 20, 40]}

    def signal(self, df):
        p = self.params
        c = df["close"]
        hi_e, lo_e = ind.donchian(df, p["entry"])
        hi_x, lo_x = ind.donchian(df, p["exit"])
        pos = _stateful(_b(c > hi_e), _b(c < lo_x), _b(c < lo_e), _b(c > hi_x), p["allow_short"])
        return pd.Series(pos, index=df.index)


@register
class Supertrend(Strategy):
    name = "supertrend"
    description = "Trend following with an ATR trailing band: long in up-trends, short in down-trends."
    defaults = {"period": 10, "mult": 3.0, "allow_short": True}
    grid = {"period": [10, 14, 21], "mult": [2.0, 3.0, 4.0]}

    def signal(self, df):
        p = self.params
        sig = ind.supertrend(df, p["period"], p["mult"])
        return sig if p["allow_short"] else sig.clip(lower=0)


@register
class TsMomentum(Strategy):
    name = "tsmom"
    description = "Time-series momentum: follow the sign of the lookback return, sized down when volatility is high."
    defaults = {"lookback": 168, "vol_window": 72, "target_vol": 0.6, "allow_short": True}
    grid = {"lookback": [72, 168, 336, 720], "target_vol": [0.4, 0.6, 0.8]}

    def signal(self, df):
        p = self.params
        c = df["close"]
        bars_per_year = YEAR_MS / infer_interval_ms(df)
        direction = np.sign(c / c.shift(p["lookback"]) - 1)
        vol = ind.realized_vol(c, p["vol_window"], bars_per_year)
        size = (p["target_vol"] / vol.replace(0, np.nan)).clip(upper=1.0)
        # Round the size to quarters so small volatility wiggles do not cause trades.
        sig = (direction * (size * 4).round() / 4).fillna(0.0)
        return sig if p["allow_short"] else sig.clip(lower=0)


@register
class FundingCarry(Strategy):
    name = "funding_carry"
    description = "Carry: short when average funding is very positive (get paid), long when very negative."
    defaults = {"window": 24, "threshold_apr": 20.0, "trend_guard": 200}
    grid = {"window": [8, 24, 72], "threshold_apr": [10.0, 20.0, 40.0], "trend_guard": [0, 200]}

    def signal(self, df):
        p = self.params
        if "funding" not in df.columns or not df["funding"].abs().sum():
            return pd.Series(0.0, index=df.index)
        bars_per_year = YEAR_MS / infer_interval_ms(df)
        apr = df["funding"].rolling(p["window"]).mean() * bars_per_year * 100
        sig = pd.Series(0.0, index=df.index)
        sig[apr > p["threshold_apr"]] = -1.0
        sig[apr < -p["threshold_apr"]] = 1.0
        if p["trend_guard"]:
            # Do not fight a strong trend just to collect funding.
            trend = ind.ema(df["close"], int(p["trend_guard"]))
            sig[(sig < 0) & (df["close"] > trend * 1.05)] = 0.0
            sig[(sig > 0) & (df["close"] < trend * 0.95)] = 0.0
        return sig


@register
class AdaptiveRegime(Strategy):
    name = "adaptive_regime"
    description = "Regime switch on ADX: follow the trend when it is strong, fade extremes when the market ranges."
    defaults = {"adx_period": 14, "trend_adx": 25, "range_adx": 18, "fast": 20, "slow": 100, "z_window": 20, "z_entry": 2.0}
    grid = {"trend_adx": [20, 25, 30], "range_adx": [15, 18], "z_entry": [1.5, 2.0, 2.5]}

    def signal(self, df):
        p = self.params
        c = df["close"]
        a = ind.adx(df, p["adx_period"])
        trend = np.sign(ind.ema(c, p["fast"]) - ind.ema(c, p["slow"]))
        z = ind.zscore(c, p["z_window"])
        revert = pd.Series(0.0, index=df.index)
        revert[z > p["z_entry"]] = -1.0
        revert[z < -p["z_entry"]] = 1.0
        sig = pd.Series(0.0, index=df.index)
        sig[a >= p["trend_adx"]] = trend[a >= p["trend_adx"]]
        sig[a <= p["range_adx"]] = revert[a <= p["range_adx"]]
        return sig.fillna(0.0)


@register
class Ensemble(Strategy):
    name = "ensemble"
    description = "Vote of the trend and momentum strategies; trades only when enough of them agree."
    defaults = {"members": "ema_cross,macd,donchian_breakout,supertrend,tsmom", "min_agreement": 0.4}
    grid = {"min_agreement": [0.2, 0.4, 0.6, 0.8]}

    def signal(self, df):
        p = self.params
        names = [m.strip() for m in str(p["members"]).split(",") if m.strip()]
        votes = pd.concat([np.sign(get_strategy(n).signal(df)) for n in names], axis=1).mean(axis=1)
        sized = (votes * 4).round() / 4
        return sized.where(votes.abs() >= p["min_agreement"], 0.0).fillna(0.0)


def tradable_strategies() -> list[str]:
    return [n for n in REGISTRY if n != "buy_hold"]
