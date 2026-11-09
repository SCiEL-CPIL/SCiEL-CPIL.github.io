# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Exponentially Discounted Expected Improvement.
    
    Strategy:
    1. Compute standard Expected Improvement (EI) with a small jitter to encourage exploration.
    2. Apply an exponential cost penalty: score = EI * exp(-lambda * cost).
    
    This differs structurally from UCB/cost by:
    - Using EI (bounded, probabilistic) instead of UCB (unbounded, heuristic).
    - Using multiplicative exponential decay instead of linear division.
    - The exponential form naturally dampens high-cost points more aggressively than linear division when costs are > 1.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety guards
    sigma_safe = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Standard Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    EI = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # Ensure EI is non-negative (numerical stability)
    EI = np.maximum(EI, 0.0)
    
    # Exponential cost penalty
    # Cost is in [1, 2]. 
    # lambda controls the strength of cost aversion.
    # If lambda is too high, we only pick the cheapest points (cost ~ 1).
    # If lambda is 0, we ignore cost.
    # A moderate value like 0.5 or 1.0 is typical.
    # Given the cost range is small [1,2], the penalty factor is exp(-lambda) to exp(-2*lambda).
    # For lambda=1.0: factors are 0.37 to 0.13. This is a strong preference for cheap points.
    # For lambda=0.5: factors are 0.61 to 0.37.
    # Let's use lambda = 0.8 to balance exploration of value vs cost.
    lambda_cost = 0.8
    
    penalty = np.exp(-lambda_cost * cost_safe)
    
    scores = EI * penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
