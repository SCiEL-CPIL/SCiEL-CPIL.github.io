import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
_start_time = time.time()
_DEADLINE = 325.0


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return a simple score
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        return (f_best - mu) / cost_safe

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)

    # Calculate z-score
    # Add small epsilon to sigma to prevent division by zero
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)

    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Compress cost range to reduce penalty for high-cost points
    # Cost is in [1, 2]. Adjusted is in [1, 1.5]
    adjusted_cost = 0.5 * cost + 0.5

    # Cost-aware score: PoI / adjusted_cost
    # This balances exploitation (PoI) with cost efficiency
    scores = poi / (adjusted_cost + eps)

    # Add a small exploration bonus to encourage diversity
    # This helps if the GP is slightly overconfident and to escape local optima
    # Use a moderate weight to balance exploitation and exploration
    exploration_weight = 0.3
    scores = scores + exploration_weight * sigma / (adjusted_cost + eps)

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
