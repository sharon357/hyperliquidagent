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


DISCRETIONARY = ("hold_long", "hold_short")


@register
class HoldLong(Strategy):
    name = "hold_long"
    description = "Always long. No rule behind it: used by the AI manager for its own bets and for hedges."

    def signal(self, df):
        return pd.Series(1.0, index=df.index)


@register
class HoldShort(Strategy):
    name = "hold_short"
    description = "Always short. No rule behind it: used by the AI manager for its own bets and for hedges."

    def signal(self, df):
        return pd.Series(-1.0, index=df.index)


def tradable_strategies() -> list[str]:
    """Rule-based strategies that get researched (not the benchmark or the constant positions)."""
    return [n for n in REGISTRY if n != "buy_hold" and n not in DISCRETIONARY]


# ---------------------------------------------------------------------------
# Strategies built on ideas from the systematic-trading literature.
# ---------------------------------------------------------------------------
def _ann_vol(df: pd.DataFrame, window: int = 72) -> pd.Series:
    return ind.realized_vol(df["close"], window, YEAR_MS / infer_interval_ms(df))


def _scaled_forecast(raw: pd.Series) -> pd.Series:
    """Rescale a raw signal so its long-run average size is 10, capped at +/-20.

    The scale uses only past values (an expanding average), so it never looks ahead.
    """
    scale = raw.abs().expanding(min_periods=50).mean().replace(0, np.nan)
    return (10.0 * raw / scale).clip(-20, 20)


def _trend_forecast(df: pd.DataFrame, speeds: tuple[int, ...]) -> pd.Series:
    """Average of several EMA-crossover forecasts, each measured in units of price volatility."""
    c = df["close"]
    price_vol = c.diff().ewm(span=36, min_periods=36).std().replace(0, np.nan)
    parts = [_scaled_forecast((ind.ema(c, f) - ind.ema(c, 4 * f)) / price_vol) for f in speeds]
    combined = pd.concat(parts, axis=1).mean(axis=1, skipna=False)
    # Averaging partly-correlated forecasts shrinks them; scale back up, then cap.
    return (combined * 1.15).clip(-20, 20)


def _carry_forecast(df: pd.DataFrame, window: int) -> pd.Series:
    """Funding carry per unit of risk: positive when being long is paid."""
    if "funding" not in df.columns or not df["funding"].abs().sum():
        return pd.Series(0.0, index=df.index)
    bars_per_year = YEAR_MS / infer_interval_ms(df)
    carry = -df["funding"].rolling(window).mean() * bars_per_year  # longs receive when funding is negative
    return _scaled_forecast(carry / _ann_vol(df).replace(0, np.nan))


def _forecast_to_position(forecast: pd.Series, df: pd.DataFrame, target_vol: float, allow_short: bool) -> pd.Series:
    """Position = conviction (forecast / 10) x volatility scaling, in quarter steps to limit churn."""
    size = (target_vol / _ann_vol(df).replace(0, np.nan)).clip(upper=1.5)
    pos = (forecast / 10.0 * size).clip(-1, 1)
    pos = ((pos * 4).round() / 4).fillna(0.0)
    return pos if allow_short else pos.clip(lower=0)


@register
class ForecastTrend(Strategy):
    name = "forecast_trend"
    description = "Several trend speeds blended into one conviction score, sized so the position targets a fixed volatility."
    defaults = {"speeds": "8,16,32,64", "target_vol": 0.5, "allow_short": True}
    grid = {"speeds": ["8,16,32,64", "16,32,64", "4,8,16,32"], "target_vol": [0.3, 0.5, 0.8]}

    def signal(self, df):
        p = self.params
        speeds = tuple(int(x) for x in str(p["speeds"]).split(","))
        return _forecast_to_position(_trend_forecast(df, speeds), df, p["target_vol"], p["allow_short"])


@register
class CarryTrend(Strategy):
    name = "carry_trend"
    description = "Blend of trend and funding carry: follow the trend, lean toward the side that gets paid funding."
    defaults = {"carry_weight": 0.4, "carry_window": 24, "target_vol": 0.5, "allow_short": True}
    grid = {"carry_weight": [0.2, 0.4, 0.6], "carry_window": [24, 72], "target_vol": [0.3, 0.5]}

    def signal(self, df):
        p = self.params
        trend = _trend_forecast(df, (8, 16, 32, 64))
        carry = _carry_forecast(df, int(p["carry_window"]))
        blend = ((1 - p["carry_weight"]) * trend + p["carry_weight"] * carry).clip(-20, 20)
        return _forecast_to_position(blend, df, p["target_vol"], p["allow_short"])


@register
class OuReversion(Strategy):
    name = "ou_reversion"
    description = "Mean reversion that only trades when prices are measurably mean-reverting (short estimated half-life)."
    defaults = {"window": 100, "max_half_life": 30, "entry": 1.5, "allow_short": True}
    grid = {"window": [50, 100, 200], "max_half_life": [15, 30, 60], "entry": [1.5, 2.0]}

    def signal(self, df):
        p = self.params
        w = int(p["window"])
        x = np.log(df["close"])
        dev = x - x.rolling(w).mean()
        lag = dev.shift(1)
        # AR(1) coefficient of the deviation: below 1 means it is pulled back to the mean.
        phi = dev.rolling(w).cov(lag) / lag.rolling(w).var().replace(0, np.nan)
        half_life = -np.log(2) / np.log(phi.where((phi > 0) & (phi < 1)))
        reverting = half_life < p["max_half_life"]
        z = dev / dev.rolling(w).std().replace(0, np.nan)
        pos = _stateful(
            _b((z < -p["entry"]) & reverting), _b(z >= 0), _b((z > p["entry"]) & reverting), _b(z <= 0), p["allow_short"]
        )
        return pd.Series(pos, index=df.index)


@register
class MetaTrend(Strategy):
    name = "meta_trend"
    description = "Trend following with a second opinion: bets only while the trend rule itself has recently been making money, sized by how well."
    defaults = {"fast": 20, "slow": 100, "window": 120, "allow_short": True}
    grid = {"fast": [10, 20], "slow": [50, 100], "window": [60, 120, 240]}

    def signal(self, df):
        p = self.params
        if p["fast"] >= p["slow"]:
            return pd.Series(0.0, index=df.index)
        c = df["close"]
        primary = np.sign(ind.ema(c, p["fast"]) - ind.ema(c, p["slow"])).fillna(0.0)
        # What the trend rule earned on each bar, using the position it held coming into the bar.
        earned = primary.shift(1) * c.pct_change()
        w = int(p["window"])
        t_stat = earned.rolling(w).mean() / earned.rolling(w).std().replace(0, np.nan) * np.sqrt(w)
        size = (t_stat / 2.0).clip(0, 1)  # zero when the rule has been losing, full at a t-stat of 2
        sig = (primary * (size * 4).round() / 4).fillna(0.0)
        return sig if p["allow_short"] else sig.clip(lower=0)
