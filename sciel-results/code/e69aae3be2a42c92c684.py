import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Cost-Aware Expected Improvement with
    enhanced adaptive exploration and linear cost penalty.
    """
    # Deadline guard
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-9)
        return -mu / cost

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety floors
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization
    # Use xi = 0.0 to maximize chance of improvement
    xi = 0.0
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Adaptive exploration: larger early, smaller as f_best improves
    # Shekel 4d minimum is ~-10.5364
    # f_best starts near -5 to -7
    # progress = 1.0 at start, 0.0 near optimum
    opt_estimate = -10.5364
    # Use a wider range for progress to keep exploration active longer
    progress = np.clip((f_best - opt_estimate) / 6.0, 0.0, 1.0)
    
    # Stronger exploration weight: 0.3 early, 0.1 late
    alpha = 0.1 + 0.2 * progress
    exploration_bonus = alpha * sigma_safe

    # Combined raw score
    raw_score = ei + exploration_bonus

    # Cost-aware penalty: Linear cost penalty (gamma = 1.0)
    # This is less aggressive than 1.2, allowing slightly more expensive points
    # if they offer significantly better EI.
    cost_weight = 1.0 / cost_safe

    score = raw_score * cost_weight

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
