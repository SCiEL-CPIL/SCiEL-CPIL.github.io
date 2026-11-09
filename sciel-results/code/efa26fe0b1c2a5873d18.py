"""Cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Wall-clock guard: the evaluator kills the program at 330s.
# Keep a safety margin so we never exceed the limit.
_T0 = time.time()
_DEADLINE = _T0 + 290.0  # 40s safety margin before the 330s kill
_EPS = 1e-9
_EXPL_W = 0.10  # weight of the intrinsic-uncertainty (exploration) bonus


def _time_left():
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (0..1) plus a bounded, decaying exploration
    # bonus, all cost-aware. Higher score -> evaluate next.
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Guard: if we are about to run out of wall-clock, return a safe finite
    # result immediately (prefer cheap, low-uncertainty points).
    if _time_left() < 0.5:
        return np.where(cost > 0, 1.0 / (cost + _EPS), 0.0)

    # Standardize sigma for a bounded, comparable exploration bonus.
    max_sigma = float(np.max(sigma)) if sigma.size else 1.0
    if not np.isfinite(max_sigma) or max_sigma <= _EPS:
        max_sigma = _EPS

    # Numerically safe z.
    z = (f_best - mu) / (sigma + _EPS)
    pi = norm.cdf(z)  # Probability of Improvement, in [0, 1]
    pi = np.clip(pi, 0.0, 1.0)

    # Intrinsic uncertainty bonus (decays as the GP becomes confident).
    exploration = (sigma / max_sigma)

    # Combine: a probability of improvement plus a small exploration floor,
    # then make it cost-aware.
    score = pi + _EXPL_W * exploration

    # Cost-aware normalization.
    safe_cost = np.where(cost > _EPS, cost, 1.0)
    score = score / safe_cost

    # Ensure the output is fully finite.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
