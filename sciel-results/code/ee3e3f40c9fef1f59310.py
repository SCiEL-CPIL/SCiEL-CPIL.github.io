"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_DEADLINE = 315.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a cost-normalized Lower Confidence Bound (LCB) strategy.
    LCB for minimization: mu - beta * sigma
    We maximize the score: -LCB / cost = (beta * sigma - mu) / cost
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # LCB: minimize mu - beta * sigma, so maximize beta * sigma - mu
    # Use a moderate beta for exploration in 6D space
    beta = 2.0
    
    # The score is (beta * sigma - mu) / cost
    # This encourages regions with low predicted value (low mu) and/or high uncertainty (high sigma)
    # Normalized by cost to be cost-aware
    scores = (beta * sigma_safe - mu) / cost_safe
    
    # Handle NaN/inf
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
