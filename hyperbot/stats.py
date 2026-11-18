"""Statistics for judging whether a backtest result is real or luck.

Two questions matter once you have tested many strategies and kept the best:
1. How likely is it that this strategy's true Sharpe ratio is above zero, given
   how short and how lumpy its track record is?  -> probabilistic Sharpe ratio
2. How likely is that still, given that it was the best of N things tried?
   -> deflated Sharpe ratio

Both follow the approach popularised by Bailey and Lopez de Prado. Sharpe ratios
here are per bar (not annualised).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
_EULER = 0.5772156649015329


def bar_sharpe(returns: pd.Series) -> float:
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else 0.0


def probabilistic_sharpe(returns: pd.Series, benchmark: float = 0.0) -> float:
    """Probability that the true per-bar Sharpe exceeds `benchmark`.

    Penalises short samples, negative skew and fat tails: a few lucky big wins
    are weaker evidence than many small ones.
    """
    return psr_from_stats(sharpe_stats(returns), benchmark)


def sharpe_stats(returns: pd.Series) -> dict:
    """The numbers the Sharpe tests need, so a result can be re-tested later without the full series."""
    r = returns.dropna()
    skew, kurt = (r.skew(), r.kurt()) if len(r) > 3 else (0.0, 0.0)
    return {
        "n": int(len(r)),
        "sr": bar_sharpe(r),
        "sd": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
        "skew": float(skew) if np.isfinite(skew) else 0.0,
        "kurt": float(kurt) + 3.0 if np.isfinite(kurt) else 3.0,  # pandas gives excess kurtosis
    }


def psr_from_stats(st: dict, benchmark: float = 0.0) -> float:
    if st["n"] < 30:
        return 0.5
    sr = st["sr"]
    denom = 1.0 - st["skew"] * sr + (st["kurt"] - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return 0.5
    return float(_N.cdf((sr - benchmark) * math.sqrt(st["n"] - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """The best per-bar Sharpe you would expect from `n_trials` worthless strategies."""
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sharpe_variance) * ((1.0 - _EULER) * a + _EULER * b))


def deflated_sharpe(returns: pd.Series, n_trials: int, sharpe_variance: float) -> float:
    """Probability the true Sharpe is above what luck alone would give the best of `n_trials`."""
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, sharpe_variance))


def half_kelly(returns: pd.Series) -> float:
    """Half the Kelly-optimal leverage (mean / variance) for a return stream. Can be <= 0."""
    r = returns.dropna()
    var = r.var(ddof=1)
    if len(r) < 30 or not var or var <= 0:
        return 0.0
    return float(0.5 * r.mean() / var)
