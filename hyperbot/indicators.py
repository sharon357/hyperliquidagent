"""Technical indicators. All are causal: the value at bar t uses only bars <= t."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = rma(delta.clip(lower=0), n)
    loss = rma(-delta.clip(upper=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def zscore(close: pd.Series, n: int = 20) -> pd.Series:
    sd = close.rolling(n).std(ddof=0)
    return (close - sma(close, n)) / sd.replace(0, np.nan)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def donchian(df: pd.DataFrame, n: int):
    """Highest high / lowest low of the previous n bars (excludes the current bar)."""
    return df["high"].rolling(n).max().shift(1), df["low"].rolling(n).min().shift(1)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_ = atr(df, n).replace(0, np.nan)
    plus_di = 100 * rma(plus_dm, n) / atr_
    minus_di = 100 * rma(minus_dm, n) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return rma(dx, n)


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    """Direction of the supertrend: +1 up-trend, -1 down-trend, 0 during warm-up."""
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    a = atr(df, n).to_numpy()
    close = df["close"].to_numpy()
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    direction = np.zeros(len(df))
    f_upper = np.full(len(df), np.nan)
    f_lower = np.full(len(df), np.nan)
    for i in range(len(df)):
        if np.isnan(a[i]):
            continue
        if i == 0 or np.isnan(f_upper[i - 1]):
            f_upper[i], f_lower[i], direction[i] = upper[i], lower[i], 1.0
            continue
        f_upper[i] = upper[i] if upper[i] < f_upper[i - 1] or close[i - 1] > f_upper[i - 1] else f_upper[i - 1]
        f_lower[i] = lower[i] if lower[i] > f_lower[i - 1] or close[i - 1] < f_lower[i - 1] else f_lower[i - 1]
        if close[i] > f_upper[i - 1]:
            direction[i] = 1.0
        elif close[i] < f_lower[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
    return pd.Series(direction, index=df.index)


def realized_vol(close: pd.Series, n: int, bars_per_year: float) -> pd.Series:
    """Annualised volatility of log returns over n bars."""
    r = np.log(close).diff()
    return r.rolling(n).std(ddof=0) * np.sqrt(bars_per_year)
