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
from itertools import combinations
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


def pbo_cscv(trial_returns: pd.DataFrame, n_blocks: int = 8) -> dict | None:
    """Probability of backtest overfitting, by combinatorially symmetric cross-validation.

    `trial_returns` has one column of per-bar returns for every parameter set that
    was tried. The history is cut into blocks; for every way of choosing half the
    blocks as "in-sample", the best in-sample parameter set is found and its rank on
    the other half is recorded. PBO is how often that winner lands in the bottom
    half out-of-sample. Near 0: tuning finds something real. 0.5 or more: picking the
    best backtest is no better than picking at random.
    """
    r = trial_returns.fillna(0.0)
    r = r.loc[:, r.std() > 0]  # parameter sets that never trade carry no information
    n = r.shape[1]
    if n < 4 or len(r) < n_blocks * 20:
        return None
    x = r.to_numpy()
    blocks = np.array_split(np.arange(len(x)), n_blocks)
    count = np.array([len(b) for b in blocks], dtype=float)
    total = np.array([x[b].sum(axis=0) for b in blocks])
    squares = np.array([(x[b] ** 2).sum(axis=0) for b in blocks])

    def sharpe(idx: list[int]) -> np.ndarray:
        m = total[idx].sum(axis=0) / count[idx].sum()
        var = squares[idx].sum(axis=0) / count[idx].sum() - m * m
        return np.nan_to_num(m / np.sqrt(np.where(var > 0, var, np.nan)))

    below_median = losses = 0
    combos = list(combinations(range(n_blocks), n_blocks // 2))
    for combo in combos:
        inside = list(combo)
        outside = [i for i in range(n_blocks) if i not in combo]
        best = int(np.argmax(sharpe(inside)))
        oos = sharpe(outside)
        rank = (oos < oos[best]).sum() + 0.5 * (oos == oos[best]).sum()  # 0..n, ties split
        below_median += rank / n <= 0.5
        losses += oos[best] < 0
    return {"pbo": below_median / len(combos), "prob_oos_loss": losses / len(combos), "configs": n, "splits": len(combos)}


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical risk parity: weights that account for correlation without inverting a noisy matrix.

    Assets are ordered so similar ones sit together (single-linkage clustering on
    correlation distance), then risk is split top-down between the two halves of each
    cluster in inverse proportion to their variance. Three near-identical bets end up
    sharing one bet's worth of weight instead of taking three.
    """
    r = returns.dropna()
    cols = list(r.columns)
    n = len(cols)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1 or len(r) < 10:
        return pd.Series(1.0 / n, index=cols)
    cov = r.cov().to_numpy().copy()  # pandas hands back a read-only view
    cov[np.diag_indices(n)] = np.maximum(np.diag(cov), 1e-12)
    dist = np.sqrt(np.clip(0.5 * (1 - r.corr().fillna(0.0).to_numpy()), 0, 1))
    # Build the cluster tree: each node is (members, left child, right child).
    nodes = [([i], None, None) for i in range(n)]
    while len(nodes) > 1:
        best = (np.inf, 0, 1)
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                d = min(dist[i, j] for i in nodes[a][0] for j in nodes[b][0])
                if d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        merged = (nodes[a][0] + nodes[b][0], nodes[a], nodes[b])
        nodes = [x for k, x in enumerate(nodes) if k not in (a, b)] + [merged]

    def cluster_var(idx: list[int]) -> float:
        sub = cov[np.ix_(idx, idx)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    # Split risk down the tree itself (not by halving a list), so a tight cluster of
    # look-alike bets is weighed as a unit against whatever it was merged with.
    w = np.ones(n)
    stack = [nodes[0]]
    while stack:
        members, left, right = stack.pop()
        if left is None:
            continue
        vl, vr = cluster_var(left[0]), cluster_var(right[0])
        alpha = 1 - vl / (vl + vr)
        w[left[0]] *= alpha
        w[right[0]] *= 1 - alpha
        stack += [left, right]
    return pd.Series(w / w.sum(), index=cols)
