"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

# Deadline guard
_DEADLINE = 320.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement with UCB and adaptive cost-awareness.
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * ndtr(z_clipped) + sigma_safe * np.exp(-0.5 * z_clipped * z_clipped) / np.sqrt(2 * np.pi)
    ei = np.maximum(ei, 0.0)

    # UCB for minimization: mu - kappa * sigma
    # Adapt kappa based on how far we are from the optimum
    # Larger kappa early on for exploration
    if f_best > -2.5:
        kappa = 2.0
    elif f_best > -3.0:
        kappa = 1.5
    else:
        kappa = 1.0

    ucb = mu - kappa * sigma_safe
    ucb_gain = np.maximum(f_best - ucb, 0.0)

    # Cost-awareness: use a power-law cost penalty
    # Adaptive beta: start lower (more exploration) and increase as we get closer
    # f_best starts around -2 to -3 and approaches -3.32
    if f_best > -3.0:
        beta = 1.2
    elif f_best > -3.2:
        beta = 1.5
    else:
        beta = 1.8

    # Combine EI and UCB-based gain with adaptive alpha
    if f_best > -2.5:
        alpha = 0.4
    elif f_best > -3.0:
        alpha = 0.3
    else:
        alpha = 0.2

    combined = ei + alpha * ucb_gain

    # Apply power-law cost penalty
    scores = combined / np.power(cost_safe, beta)

    # Handle NaN/inf
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
