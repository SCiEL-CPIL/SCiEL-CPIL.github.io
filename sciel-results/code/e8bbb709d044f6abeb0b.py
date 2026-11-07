"""Cost-aware acquisition for Bayesian optimization of powell 4d (min 0) on [-4,5]^4."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_EPS = 1e-9
_DEADLINE_S = 330.0
_SAFETY_S = 2.0


def _expected_improvement(mu, sigma, f_best):
    """Standard Expected Improvement with f_best as target."""
    improvement = f_best - mu
    sigma_safe = np.maximum(sigma, _EPS)
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei


def acquisition(mu, sigma, f_best, cost):
    t0 = time.time()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite inputs.
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, _EPS)
    sigma = np.clip(sigma, _EPS, None)
    cost = np.where(np.isfinite(cost), cost, 1.5)
    cost = np.clip(cost, _EPS, None)

    # Core: Expected Improvement
    ei = _expected_improvement(mu, sigma, f_best)

    # Cost-aware: EI divided by cost
    score = ei / cost

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - t0 > (_DEADLINE_S - _SAFETY_S):
        return score

    return score
# EVOLVE-BLOCK-END
