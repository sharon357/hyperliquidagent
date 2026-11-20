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
    intervals: tuple | None = None  # candle sizes the rule makes sense on; None = any
    research = True  # False: kept for manual backtests, but the autopilot neither researches nor trades it

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
    return [n for n, cls in REGISTRY.items() if n != "buy_hold" and n not in DISCRETIONARY and cls.research]


# On the day-trading sleeve's short candles only these are researched. In a walk-forward test on
# eight coins (52 days of 15m candles, costs included) the trend and band rules lost on every coin
# because they trade too often for their edge; these were the ones that did not.
DAY_STRATEGIES = ("funding_carry", "carry_trend", "rsi_reversion", "tsmom", "vol_breakout", "nday_high", "trend_carry")


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


# ---------------------------------------------------------------------------
# Strategies taken from published research write-ups (Quantpedia, Quant Memo,
# the awesome-systematic-trading collection, QuantifiedTrader) and from widely
# published short-term trading rules. Lookbacks are given in days or hours and
# converted to bars, so the same rule works on 15m, 1h and 4h candles.
# ---------------------------------------------------------------------------
INTRADAY_INTERVALS = ("5m", "15m", "30m", "1h")


def _per_day(df: pd.DataFrame) -> int:
    return max(int(round(86_400_000 / infer_interval_ms(df))), 1)


def _daily_atr(df: pd.DataFrame, days: int = 14) -> pd.Series:
    """A typical day's range, estimated from intraday bars (bar ATR scaled by the square root of bars per day)."""
    per_day = _per_day(df)
    return ind.atr(df, max(days * per_day // 4, 14)) * np.sqrt(per_day)


@register
class SessionWindow(Strategy):
    name = "session_window"
    description = "Time-of-day effect: hold a position only during a fixed window of UTC hours each day (published for Bitcoin as long 22:00-00:00 UTC)."
    research = False  # lost after costs on 8 coins x 3 timeframes (walk-forward, Nov 2026)
    defaults = {"start_hour": 22, "hold_hours": 2, "side": 1}
    grid = {"start_hour": [20, 21, 22, 23], "hold_hours": [2, 3, 4]}
    intervals = INTRADAY_INTERVALS

    def signal(self, df):
        p = self.params
        # The value at bar t is the position for the NEXT bar, so test where the next bar starts.
        nxt = df.index + pd.Timedelta(milliseconds=infer_interval_ms(df))
        minutes = (nxt.hour * 60 + nxt.minute - int(p["start_hour"]) * 60) % 1440
        inside = minutes < int(p["hold_hours"]) * 60
        return pd.Series(np.where(inside, float(np.sign(p["side"])), 0.0), index=df.index)


@register
class MtfMacd(Strategy):
    name = "mtf_macd"
    description = "Multi-timeframe trend: trade intraday MACD crossovers only in the direction of the daily MACD, and leave on the first bar that closes against the trade."
    research = False  # lost after costs on 8 coins x 3 timeframes (walk-forward, Nov 2026)
    defaults = {"fast": 12, "slow": 26, "signal": 9, "max_bars": 0, "allow_short": True}
    grid = {"slow": [26, 52], "max_bars": [0, 1, 4]}

    def signal(self, df):
        p = self.params
        c = df["close"]
        # Daily trend from completed days only: yesterday's reading is what is known during today.
        daily = c.resample("1D").last().dropna()
        d_line, d_sig, _ = ind.macd(daily, p["fast"], p["slow"], p["signal"])
        d_dir = np.sign(d_line - d_sig).shift(1)
        day_dir = d_dir.reindex(df.index.floor("1D")).to_numpy()
        line, sig, _ = ind.macd(c, p["fast"], p["slow"], p["signal"])
        above = (line > sig).to_numpy()
        prev = np.roll(above, 1)
        prev[0] = above[0]
        cross_up, cross_dn = above & ~prev, ~above & prev
        green = (c >= df["open"]).to_numpy()
        n, out, pos, held = len(df), np.zeros(len(df)), 0.0, 0
        limit = int(p["max_bars"])
        for i in range(n):
            if pos != 0:
                held += 1
                against = (pos > 0 and not green[i]) or (pos < 0 and green[i])
                if against or (limit and held >= limit) or day_dir[i] != pos:
                    pos = 0.0
            if pos == 0:
                if cross_up[i] and day_dir[i] > 0:
                    pos, held = 1.0, 0
                elif cross_dn[i] and day_dir[i] < 0 and p["allow_short"]:
                    pos, held = -1.0, 0
            out[i] = pos
        return pd.Series(out, index=df.index)


@register
class NDayHigh(Strategy):
    name = "nday_high"
    description = "Price-extreme momentum: long for a set time after a close at an N-day high (published for Bitcoin with N=10 to 20), short after an N-day low."
    defaults = {"days": 10, "hold_days": 1, "allow_short": True}
    grid = {"days": [10, 20, 30], "hold_days": [1, 2, 3]}

    def signal(self, df):
        p = self.params
        per_day = _per_day(df)
        n, hold = int(p["days"]) * per_day, max(int(p["hold_days"] * per_day), 1)
        c = df["close"]
        at_high = (c >= c.shift(1).rolling(n).max()).astype(float)
        at_low = (c <= c.shift(1).rolling(n).min()).astype(float)
        long_on = at_high.rolling(hold, min_periods=1).max()
        short_on = at_low.rolling(hold, min_periods=1).max() if p["allow_short"] else 0.0
        return (long_on - short_on).fillna(0.0).clip(-1, 1)


@register
class VolBreakout(Strategy):
    name = "vol_breakout"
    description = "Range expansion: enter when price closes beyond its N-day range by a fraction of a day's range, exit on a trailing stop measured in daily ranges."
    defaults = {"days": 20, "atr_frac": 0.25, "trail_atr": 3.0, "allow_short": True}
    grid = {"days": [10, 20, 55], "trail_atr": [2.0, 3.0]}

    def signal(self, df):
        p = self.params
        n = int(p["days"]) * _per_day(df)
        c = df["close"].to_numpy()
        hi = df["high"].shift(1).rolling(n).max().to_numpy()
        lo = df["low"].shift(1).rolling(n).min().to_numpy()
        day = _daily_atr(df).to_numpy()
        out, pos, best = np.zeros(len(c)), 0.0, 0.0
        for i in range(len(c)):
            if not np.isfinite(hi[i]) or not np.isfinite(day[i]):
                continue
            if pos > 0:
                best = max(best, c[i])
                if c[i] < best - p["trail_atr"] * day[i]:
                    pos = 0.0
            elif pos < 0:
                best = min(best, c[i])
                if c[i] > best + p["trail_atr"] * day[i]:
                    pos = 0.0
            if pos == 0:
                if c[i] > hi[i] + p["atr_frac"] * day[i]:
                    pos, best = 1.0, c[i]
                elif p["allow_short"] and c[i] < lo[i] - p["atr_frac"] * day[i]:
                    pos, best = -1.0, c[i]
            out[i] = pos
        return pd.Series(out, index=df.index)


@register
class AtrBands(Strategy):
    name = "atr_bands"
    description = "Mean reversion: buy a close at or below the moving average minus a multiple of ATR, exit back at the average (mirror image for shorts)."
    research = False  # lost after costs on 8 coins x 3 timeframes (walk-forward, Nov 2026)
    defaults = {"period": 14, "mult": 2.0, "allow_short": True}
    grid = {"period": [14, 20, 50], "mult": [1.5, 2.0, 2.5, 3.0]}

    def signal(self, df):
        p = self.params
        c = df["close"]
        mid = ind.sma(c, int(p["period"]))
        band = p["mult"] * ind.atr(df, int(p["period"]))
        pos = _stateful(_b(c <= mid - band), _b(c >= mid), _b(c >= mid + band), _b(c <= mid), p["allow_short"])
        return pd.Series(pos, index=df.index)


@register
class OpeningRange(Strategy):
    name = "opening_range"
    description = "Day trade: mark the high and low of the first hours of a session, take the first close outside that range, and be flat again by the end of the session."
    research = False  # lost after costs on 8 coins x 3 timeframes (walk-forward, Nov 2026)
    defaults = {"session_start": 0, "range_hours": 1, "allow_short": True}
    grid = {"session_start": [0, 8, 13], "range_hours": [1, 2, 4]}
    intervals = INTRADAY_INTERVALS

    def signal(self, df):
        p = self.params
        step = infer_interval_ms(df) // 60_000  # minutes per bar
        start = pd.Timedelta(hours=int(p["session_start"]))
        shifted = df.index - start
        session = shifted.floor("1D").asi8  # which session each bar belongs to
        minute = (shifted.hour * 60 + shifted.minute).to_numpy()  # minutes since the session opened
        range_end = int(p["range_hours"]) * 60
        h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
        out = np.zeros(len(c))
        cur, r_hi, r_lo, pos, traded = None, -np.inf, np.inf, 0.0, False
        for i in range(len(c)):
            if session[i] != cur:
                cur, r_hi, r_lo, pos, traded = session[i], -np.inf, np.inf, 0.0, False
            if minute[i] < range_end:
                r_hi, r_lo = max(r_hi, h[i]), min(r_lo, l[i])  # still building the range
            elif minute[i] + step >= 1440:
                pos = 0.0  # last bar of the session: flat for the next one
            else:
                if pos > 0 and c[i] < r_lo or pos < 0 and c[i] > r_hi:
                    pos = 0.0  # back through the far side of the range: the breakout failed
                if pos == 0 and not traded and np.isfinite(r_hi):
                    if c[i] > r_hi:
                        pos, traded = 1.0, True
                    elif p["allow_short"] and c[i] < r_lo:
                        pos, traded = -1.0, True
            out[i] = pos
        return pd.Series(out, index=df.index)


@register
class Rsi2Pullback(Strategy):
    name = "rsi2_pullback"
    description = "Short-term pullback: buy a very oversold 2-period RSI while price is above its long moving average, exit on a close back above a short average (mirror image in downtrends)."
    defaults = {"entry": 10, "trend": 200, "exit_ma": 5, "allow_short": True}
    grid = {"entry": [5, 10, 15], "trend": [100, 200], "exit_ma": [5, 10]}

    def signal(self, df):
        p = self.params
        c = df["close"]
        r = ind.rsi(c, 2)
        trend, fast = ind.sma(c, int(p["trend"])), ind.sma(c, int(p["exit_ma"]))
        pos = _stateful(
            _b((r < p["entry"]) & (c > trend)), _b(c > fast), _b((r > 100 - p["entry"]) & (c < trend)), _b(c < fast), p["allow_short"]
        )
        return pd.Series(pos, index=df.index)


@register
class IbsReversion(Strategy):
    name = "ibs_reversion"
    description = "Internal bar strength: buy after a bar that closes near its low, sell after one that closes near its high."
    research = False  # lost after costs on 8 coins x 3 timeframes (walk-forward, Nov 2026)
    defaults = {"low": 0.2, "exit": 0.5, "smooth": 1, "allow_short": True}
    grid = {"low": [0.1, 0.2, 0.3], "exit": [0.5, 0.7], "smooth": [1, 3]}

    def signal(self, df):
        p = self.params
        span = (df["high"] - df["low"]).replace(0, np.nan)
        ibs = ((df["close"] - df["low"]) / span).rolling(int(p["smooth"])).mean()
        pos = _stateful(
            _b(ibs < p["low"]), _b(ibs > p["exit"]), _b(ibs > 1 - p["low"]), _b(ibs < 1 - p["exit"]), p["allow_short"]
        )
        return pd.Series(pos, index=df.index)


# ---------------------------------------------------------------------------
# Rules distilled from consistently profitable Hyperliquid wallets (a study of 40
# leaderboard accounts, Nov 2026): swing traders faded 24h extremes and added to
# winners, day traders bought US-session breakouts of the 24h range, and the
# steadiest books stayed short the falling coins that paid them funding. Only the
# last survived walk-forward testing; the others are kept for the record.
# ---------------------------------------------------------------------------


@register
class FadeExtreme(Strategy):
    name = "fade_extreme"
    research = False  # lost over 1,000 daily bars (5 of 17 coins up) and on 1h; positive only on 4h in a falling half-year (walk-forward, Nov 2026)
    description = "Swing fade of a 24h extreme: buy a 24h drop of N% that closes near its low (short the mirror), unless the week is breaking down; add once in profit, trail a winner, 14-day time stop."
    defaults = {"move_pct": 3.0, "range_frac": 0.2, "week_guard_pct": 10.0, "stop_atr": 2.5, "trail_atr": 3.0, "max_days": 14, "allow_short": True}
    grid = {"move_pct": [3.0, 5.0], "stop_atr": [2.0, 3.0]}

    def signal(self, df):
        p = self.params
        n = _per_day(df)
        c = df["close"]
        r24 = c / c.shift(n) - 1
        r168 = c / c.shift(7 * n) - 1
        hi, lo = df["high"].rolling(n).max(), df["low"].rolling(n).min()
        where = (c - lo) / (hi - lo).replace(0, np.nan)
        atr = _daily_atr(df).to_numpy()
        mv, fr, wg = p["move_pct"] / 100, p["range_frac"], p["week_guard_pct"] / 100
        long_in = ((r24 <= -mv) & (where <= fr) & (r168 > -wg)).to_numpy()
        short_in = ((r24 >= mv) & (where >= 1 - fr) & (r168 < wg)).to_numpy() if p["allow_short"] else np.zeros(len(c), bool)
        px = c.to_numpy()
        out = np.zeros(len(px))
        pos, entry, stop, best, added, age = 0.0, 0.0, 0.0, 0.0, False, 0
        limit = int(p["max_days"] * n)
        for i in range(len(px)):
            a = atr[i]
            if not np.isfinite(a) or a <= 0:
                continue
            if pos:
                side = np.sign(pos)
                age += 1
                best = max(best, px[i]) if side > 0 else min(best, px[i])
                if side * (best - entry) >= 2 * a:  # trade is working: trail it
                    trail = best - side * p["trail_atr"] * a
                    stop = max(stop, trail) if side > 0 else min(stop, trail)
                if not added and side * (px[i] - entry) >= 1.5 * a:
                    pos, added = side * 1.0, True  # add to the winner, never to a loser
                if (side > 0 and px[i] <= stop) or (side < 0 and px[i] >= stop) or age >= limit:
                    pos = 0.0
            if not pos:
                if long_in[i]:
                    pos, entry, best, added, age = 2 / 3, px[i], px[i], False, 0
                    stop = px[i] - p["stop_atr"] * a
                elif short_in[i]:
                    pos, entry, best, added, age = -2 / 3, px[i], px[i], False, 0
                    stop = px[i] + p["stop_atr"] * a
            out[i] = pos
        return pd.Series(out, index=df.index)


@register
class TrendCarry(Strategy):
    name = "trend_carry"
    description = "Trend side that is paid funding: short a falling coin (7-day return negative, below its 7-day average) while longs pay funding, adding into rallies; long mirror optional. Exit on a new 7-day extreme against it, a reversal of the week, or funding flipping."
    defaults = {"side": -1, "use_funding": True, "add_pct": 3.0, "exit_week_pct": 5.0}
    grid = {"side": [-1, 0, 1], "use_funding": [True, False]}

    def signal(self, df):
        p = self.params
        n = _per_day(df)
        c = df["close"]
        r24 = (c / c.shift(n) - 1).to_numpy()
        r168 = (c / c.shift(7 * n) - 1).to_numpy()
        avg = c.rolling(7 * n).mean().to_numpy()
        hi = df["high"].shift(1).rolling(7 * n).max().to_numpy()
        lo = df["low"].shift(1).rolling(7 * n).min().to_numpy()
        f = df["funding"] if "funding" in df.columns else pd.Series(0.0, index=df.index)
        fund = f.rolling(n).mean().to_numpy()
        px = c.to_numpy()
        sides = (-1, 1) if int(p["side"]) == 0 else (int(p["side"]),)
        add, wk = p["add_pct"] / 100, p["exit_week_pct"] / 100
        out = np.zeros(len(px))
        pos, units, last_add = 0, 0, -1
        for i in range(len(px)):
            if not (np.isfinite(avg[i]) and np.isfinite(hi[i]) and np.isfinite(r168[i])):
                continue
            paid = {-1: fund[i] > 0, 1: fund[i] < 0} if p["use_funding"] else {-1: True, 1: True}
            if pos:
                against = px[i] > hi[i] if pos < 0 else px[i] < lo[i]
                week_turned = (r168[i] > wk) if pos < 0 else (r168[i] < -wk)
                if against or week_turned or (p["use_funding"] and not paid[pos] and fund[i] != 0):
                    pos, units = 0, 0
                elif units < 3 and i - last_add >= n and pos * r24[i] <= -add and (px[i] < avg[i] if pos < 0 else px[i] > avg[i]):
                    units, last_add = units + 1, i  # add into the counter-move, still on the right side of the average
            if not pos:
                for s in sides:
                    trend_ok = (r168[i] < 0 and px[i] < avg[i]) if s < 0 else (r168[i] > 0 and px[i] > avg[i])
                    if trend_ok and paid[s]:
                        pos, units, last_add = s, 1, i
                        break
            out[i] = pos * units / 3
        return pd.Series(out, index=df.index)


@register
class SessionBreakout(Strategy):
    name = "session_breakout"
    research = False  # about break-even on 15m (4 of 8 coins) and lost on 1h (3 of 8) after costs (walk-forward, Nov 2026)
    description = "US-session breakout: on bars closing 12:00-20:00 UTC, enter when price closes beyond the prior 24h high or low; stop at the range midpoint, add once in profit, take profit at 1.5 day-ranges, exit back inside the range or after 24h."
    defaults = {"start_hour": 12, "end_hour": 20, "target_atr": 1.5, "max_hours": 24, "allow_short": True}
    grid = {"target_atr": [1.0, 1.5, 2.5], "max_hours": [12, 24]}
    intervals = INTRADAY_INTERVALS

    def signal(self, df):
        p = self.params
        n = _per_day(df)
        c = df["close"].to_numpy()
        hi = df["high"].shift(1).rolling(n).max().to_numpy()
        lo = df["low"].shift(1).rolling(n).min().to_numpy()
        atr = _daily_atr(df).to_numpy()
        close_time = df.index + pd.Timedelta(hours=24 / n)
        hours = close_time.hour.to_numpy()
        in_session = (hours >= int(p["start_hour"])) & (hours <= int(p["end_hour"]))
        limit = max(int(p["max_hours"] * n / 24), 1)
        out = np.zeros(len(c))
        pos, entry, stop, level, age, added = 0.0, 0.0, 0.0, 0.0, 0, False
        for i in range(len(c)):
            a = atr[i]
            if not (np.isfinite(hi[i]) and np.isfinite(a) and a > 0):
                continue
            if pos:
                side = np.sign(pos)
                age += 1
                if not added and side * (c[i] - entry) >= 0.5 * a:
                    pos, added = side * 1.0, True
                hit_stop = c[i] <= stop if side > 0 else c[i] >= stop
                target = side * (c[i] - entry) >= p["target_atr"] * a
                back_in = c[i] < level if side > 0 else c[i] > level
                if hit_stop or target or back_in or age >= limit:
                    pos = 0.0
            if not pos and in_session[i]:
                mid = (hi[i] + lo[i]) / 2
                if c[i] > hi[i]:
                    pos, entry, stop, level, age, added = 2 / 3, c[i], mid, hi[i], 0, False
                elif p["allow_short"] and c[i] < lo[i]:
                    pos, entry, stop, level, age, added = -2 / 3, c[i], mid, lo[i], 0, False
            out[i] = pos
        return pd.Series(out, index=df.index)
