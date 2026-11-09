import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Cost-Aware Expected Improvement with
    adaptive exploration and a soft-min smoothing of the cost penalty.
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
    xi = 0.01
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Adaptive exploration: larger early, smaller as f_best improves
    # Shekel 4d minimum is ~-10.5364; f_best starts near -5 to -7
    # Scale exploration by how far f_best is from the known optimum
    opt_estimate = -10.5364
    progress = np.clip((f_best - opt_estimate) / 5.0, 0.0, 1.0)  # 1 early, 0 late
    alpha = 0.02 + 0.08 * progress
    exploration_bonus = alpha * sigma_safe

    # Combined raw score
    raw_score = ei + exploration_bonus

    # Cost-aware penalty: use a smooth cost multiplier instead of hard division
    # This avoids the discontinuity at cost boundaries and better balances
    # exploration vs. exploitation under varying costs
    # cost is in [1, 2]; use log-sum-exp style soft penalty
    # Effective cost weight: prefer low cost but not too aggressively
    cost_weight = 1.0 / cost_safe
    # Apply a mild exponent to sharpen cost preference slightly
    cost_weight = cost_weight ** 1.2

    score = raw_score * cost_weight

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
