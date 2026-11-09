"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 10.0 # Seconds before the 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: The lowest observed objective value.
        cost: Array of shape (N,) with evaluation costs.
        
    Returns:
        Array of shape (N,) with acquisition scores. Higher is better.
    """
    # Check deadline
    if time.time() - _START_TIME > 330.0 - DEADLINE_MARGIN:
        # Fallback: Maximize -mu / cost (exploitation with cost awareness)
        safe_cost = np.maximum(cost, 1e-6)
        return -mu / safe_cost

    # Probability of Improvement (PI)
    # P(y < f_best) where y ~ N(mu, sigma^2)
    # z = (f_best - mu) / sigma
    # P = Phi(z)
    
    # Handle sigma = 0 cases
    safe_sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / safe_sigma
    
    # Probability of improvement
    pi = norm.cdf(z)
    
    # Cost-aware scoring: divide by cost to prefer cheaper evaluations
    # We want to maximize the probability of improvement per unit cost
    safe_cost = np.maximum(cost, 1e-6)
    scores = pi / safe_cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
