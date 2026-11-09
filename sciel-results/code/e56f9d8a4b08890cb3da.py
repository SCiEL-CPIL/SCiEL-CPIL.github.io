"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 15.0


def _get_remaining_time():
    """Return remaining time before the deadline, with a safety margin."""
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    array of shape (2000,)
        Acquisition scores (higher = better)
    """
    # Deadline guard
    if _get_remaining_time() < 0:
        return np.zeros(2000, dtype=np.float64)

    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Handle edge cases
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Compute Upper Confidence Bound (UCB) term for exploration
    # Use a moderate exploration parameter
    xi = 0.1  # exploration-exploitation trade-off
    ucb = mu + xi * sigma

    # For minimization, we want low values, so UCB score = -mu + xi*sigma
    # But we need to combine with EI in a cost-aware way

    # Cost-aware EI
    ei_cost = ei / cost

    # Cost-aware exploration: favor points with high uncertainty relative to cost
    # Use sigma/cost as an exploration bonus
    exploration_bonus = sigma / cost

    # Combine EI and exploration with a weight
    # The weight adapts based on how good f_best is relative to the scale
    # When f_best is close to 0 (good), we can be more exploitative
    # When f_best is large, we need more exploration
    scale = max(abs(f_best), 1e-6)
    # Adaptive weight: more exploration when f_best is large
    w_explore = 0.3 * min(1.0, scale / 10.0)

    scores = ei_cost + w_explore * exploration_bonus

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # If any non-finite values remain, replace with 0
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
