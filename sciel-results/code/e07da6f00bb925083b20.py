# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Cost-Adjusted Upper Confidence Bound (UCB) for minimization.
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero and negative sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # UCB parameter: controls exploration vs exploitation
    # Higher beta encourages exploring high uncertainty regions
    beta = 2.0
    
    # For minimization, we want to find low values.
    # The "optimistic" lower bound is mu - beta * sigma.
    # We want to minimize this lower bound.
    # To maximize the score (higher = better), we negate the lower bound.
    # lower_bound = mu - beta * sigma
    # score_raw = -lower_bound = beta * sigma - mu
    
    # Normalize by cost
    scores = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
