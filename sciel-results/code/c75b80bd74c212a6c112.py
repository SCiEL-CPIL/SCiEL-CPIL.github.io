"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes the acquisition score for each candidate point.
    Uses UCB (Upper Confidence Bound) for minimization.
    
    Args:
        mu: (N,) array of posterior means.
        sigma: (N,) array of posterior standard deviations.
        f_best: float, the best (lowest) observed value so far.
        
    Returns:
        (N,) array of acquisition scores. Higher is better.
    """
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # UCB for minimization: score = -mu + beta * sigma
    # Higher score means more promising (lower expected value or high uncertainty)
    beta = 2.0
    scores = -mu + beta * sigma_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
