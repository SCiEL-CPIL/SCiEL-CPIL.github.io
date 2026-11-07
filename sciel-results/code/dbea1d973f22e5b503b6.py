import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses UCB with soft cost discounting.
    Returns 2000 finite scores (higher = evaluate next).
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Hyperparameters
    beta = 2.5  # Exploration-exploitation trade-off
    alpha = 0.7 # Soft cost discounting exponent
    
    # Calculate UCB: mu + beta * sigma
    ucb = mu + beta * sigma
    
    # Apply soft cost discounting
    # Lower cost -> higher score. 
    # cost is in [1, 2]. cost^-0.7 is in [2^-0.7, 1] ~ [0.615, 1.0]
    cost_factor = np.power(cost, -alpha)
    
    scores = ucb * cost_factor
    
    # Deadline guard
    # If this takes too long, return a safe default (e.g., just UCB without cost or zeros)
    # But for 2000 points, this should be < 1ms. 
    # The guard is more for safety against pathological inputs or hangs.
    if time.time() - t_start > 2.0: 
         # Fallback: return normalized UCB
         return ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
