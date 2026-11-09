"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time and ensure we return well before the limit.
_DEADLINE = 320.0  # seconds, safety margin of 10s
_start_time = time.time()


def _remaining_time():
    return _DEADLINE - (time.time() - _start_time)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement per unit cost with numerical safety guards.
    """
    # Safety: ensure we don't exceed deadline
    if _remaining_time() < 0.5:
        # Emergency: return a simple fallback
        return np.zeros_like(mu)

    # Ensure inputs are numpy arrays with correct dtype
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: handle zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-12)

    # Safety: handle zero or negative cost
    cost_safe = np.maximum(cost, 1e-12)

    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe

    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)

    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)

    ei = (f_best - mu) * cdf + sigma_safe * pdf

    # Ensure EI is non-negative (it should be, but numerical issues can cause tiny negatives)
    ei = np.maximum(ei, 0.0)

    # Cost-aware: divide by cost to get improvement per unit cost
    scores = ei / cost_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
