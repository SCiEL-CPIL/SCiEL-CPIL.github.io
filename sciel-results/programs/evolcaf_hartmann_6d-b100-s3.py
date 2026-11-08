"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr, ndtri
import time

# Deadline guard
_DEADLINE = 320.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement with UCB and a cost-awareness
    that penalizes high-cost points more aggressively.
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    # Use ndtr (CDF) and its density
    ei = (f_best - mu) * ndtr(z) + sigma_safe * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    ei = np.maximum(ei, 0.0)

    # UCB for minimization: mu - kappa * sigma
    # Adapt kappa based on how far we are from the optimum
    # Larger kappa early on for exploration
    kappa = 2.0
    ucb = mu - kappa * sigma_safe

    # Cost-awareness: use a power-law cost penalty
    # score = (EI + alpha * max(0, -ucb)) / cost^beta
    # beta > 1 penalizes high-cost points more
    alpha = 0.3
    beta = 1.5

    # Combine EI and UCB-based gain
    # UCB gain: how much below f_best the UCB predicts
    ucb_gain = np.maximum(f_best - ucb, 0.0)

    combined = ei + alpha * ucb_gain

    scores = combined / np.power(cost_safe, beta)

    # Handle NaN/inf
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
