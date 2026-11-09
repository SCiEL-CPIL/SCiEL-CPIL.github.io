"""Rewritten cost-aware acquisition for Bayesian optimization (4 test functions)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 330.0 - 25.0  # safety margin before the 330 s wall-clock kill
_T0 = time.perf_counter()


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware expected-improvement acquisition (higher = evaluate next).

    mu, sigma, cost: 1-D arrays of shape (N,). Returns N finite scores.
    """
    t0 = time.perf_counter()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite / non-positive inputs (defensive; should not trigger).
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 0.0)

    # Numerically safe Expected Improvement (f_best - mu) is the improvement.
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Where uncertainty is negligible, improvement is ~0; force exactly 0.
    ei = np.where(sigma > 1e-9, ei, 0.0)
    ei = np.maximum(ei, 0.0)

    # Cost-aware ranking: improvement per unit cost, plus a tiny exploration term.
    score = ei / cost + 1e-4 * sigma / cost

    # Ensure all outputs are finite.
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard: return the best (finite) result found before the limit.
    if (time.perf_counter() - t0) > (_DEADLINE - (time.perf_counter() - _T0)):
        return score
    return score
# EVOLVE-BLOCK-END
