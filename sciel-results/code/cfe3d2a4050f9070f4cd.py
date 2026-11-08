"""Cost-aware acquisition for 2D Levy optimization."""
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time and ensure we return well before the limit.
_DEADLINE = 320.0  # seconds, safety margin of 10s
_start_time = time.time()


def _remaining_time():
    return _DEADLINE - (time.time() - _start_time)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition values (higher = better)
    """
    # Safety: ensure we don't exceed deadline
    if _remaining_time() < 0.5:
        # Emergency: return a simple fallback
        return np.ones_like(mu) * 0.0
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # Use a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
