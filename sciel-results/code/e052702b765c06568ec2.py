"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard state
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    if _check_deadline():
        return np.zeros(len(mu), dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # UCB: Upper Confidence Bound
    # Higher beta encourages exploration of high-uncertainty areas
    # f_best - mu is the "optimistic improvement" if the true value is at the upper bound
    # We add beta * sigma to account for uncertainty
    beta = 1.5
    
    ucb = (f_best - mu) + beta * sigma_safe
    
    # Cost-aware: divide by cost
    # We want high UCB and low cost
    scores = ucb / cost
    
    # Ensure all scores are finite and non-negative (optional, but good for stability)
    # UCB can be negative if mu > f_best + beta*sigma, which means the point is likely worse than best.
    # However, dividing by cost (positive) preserves sign.
    # If scores are negative, they are less attractive than 0.
    # We replace non-finite with 0.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
