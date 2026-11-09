"""Cost-aware acquisition for Bayesian optimization with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # 5s safety margin before 300s limit

def _time_remaining():
    """Return seconds remaining before hard deadline."""
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB with cost normalization.
    
    Uses a hybrid approach:
    1. UCB (Upper Confidence Bound) for exploration: -mu + beta * sigma
    2. Cost-aware normalization: divide by cost to favor cheaper points
    3. Adaptive beta based on optimization progress (sigma relative to f_best)
    
    The beta parameter adapts: higher exploration early, more exploitation late.
    """
    # Time guard: if we're running low, return a simple safe answer
    if _time_remaining() < 2.0:
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
        mu_arr = np.asarray(mu, dtype=np.float64)
        cost_arr = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        return (-mu_arr + 2.0 * sigma_safe) / cost_arr

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive and cost is safe for division
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Adaptive beta: balance exploration/exploitation based on progress
    # When sigma is large relative to f_best, we're early -> explore more
    # When sigma is small relative to f_best, we're late -> exploit more
    sigma_mean = np.mean(sigma_safe)
    if abs(f_best) > 1e-6:
        sigma_rel = sigma_mean / abs(f_best)
    else:
        sigma_rel = 1.0
    
    # Clamp sigma_rel to reasonable range
    sigma_rel = np.clip(sigma_rel, 0.0, 5.0)
    
    # Beta interpolates from 3.0 (early, high exploration) to 1.0 (late, high exploitation)
    beta = 3.0 - 2.0 * np.clip(sigma_rel / 3.0, 0.0, 1.0)

    # UCB term: balances exploitation (mu) and exploration (sigma)
    # For minimization, we want low mu and high sigma
    # So we use: -mu + beta * sigma (negate mu since lower is better)
    ucb = -mu + beta * sigma_safe

    # Cost-aware: divide by cost to favor cheaper evaluations
    # This makes the acquisition function prefer points that are both promising
    # and cheap to evaluate
    cost_aware = ucb / cost_safe

    # Ensure finite output
    score = np.where(np.isfinite(cost_aware), cost_aware, 0.0)
    
    return score
# EVOLVE-BLOCK-END
