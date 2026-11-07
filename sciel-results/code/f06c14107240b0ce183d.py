"""Cost-aware acquisition for Griewank-2D BO with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Wall-clock deadline guard (330 s kill, use 325 s safety margin)
_T0 = time.time()
_DEADLINE = 325.0


def _remaining():
    """Return seconds remaining before deadline; 0 if exceeded."""
    return max(0.0, _DEADLINE - (time.time() - _T0))


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for minimizing Griewank-2D.

    Combines expected improvement with a UCB-style uncertainty bonus,
    normalized by a power of the evaluation cost to better balance
    exploration vs. exploitation under a cost budget.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean in objective units.
    sigma : ndarray, shape (2000,)
        GP posterior std in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.

    Returns
    -------
    ndarray, shape (2000,)
        Finite scores; higher = more worth evaluating next.
    """
    # Guard: if we're near deadline, return a safe finite array immediately
    if _remaining() < 0.5:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clamp sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is non-negative by construction, but clamp for safety
    ei = np.maximum(ei, 0.0)

    # UCB-style uncertainty bonus (beta controls exploration)
    beta = 1.5
    ucb_bonus = beta * sigma_safe

    # Combined acquisition: EI + uncertainty bonus
    combined = ei + ucb_bonus

    # Power-based cost normalization: higher power penalizes expensive evals more
    # Cost range is [1, 2], so cost^1.5 amplifies the penalty for high-cost regions
    cost_power = 1.5
    score = combined / (cost_safe ** cost_power)

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
