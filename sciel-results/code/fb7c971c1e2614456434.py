"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using GP-UCB with a softer cost penalty.
    
    Parameters:
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : numpy array, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Add small noise floor to sigma to prevent division by zero
    sigma_safe = sigma + 1e-10
    
    # GP-UCB acquisition: f_best + beta * sigma
    # beta is the exploration parameter. 
    # Based on inspirations, a value around 0.25-0.3 is effective.
    beta = 0.3
    
    ucb = f_best + beta * sigma_safe
    
    # Cost-aware: Penalize high costs.
    # Instead of linear division (which can be very harsh), we use a softer penalty.
    # The cost is typically in range [1, 2] for this problem (1 + mean(x scaled to [0,1])).
    # Using sqrt(cost) or 1/sqrt(cost) reduces the penalty for expensive points compared to 1/cost.
    # Let's use a multiplicative factor: score = ucb / sqrt(cost)
    # This is equivalent to ucb * cost^{-0.5}
    
    cost_safe = np.maximum(cost, 1e-10)
    
    # We want higher scores to be better.
    # ucb is higher when we are close to f_best (exploitation) or high uncertainty (exploration).
    # We divide by sqrt(cost) to prefer cheaper points, but not as aggressively as dividing by cost.
    
    scores = ucb / np.sqrt(cost_safe)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
