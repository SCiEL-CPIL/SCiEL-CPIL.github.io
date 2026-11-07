"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement with an exploration bonus (UCB-like term).
    # The exploration term helps escape local minima in the high-dimensional space.
    # 0.5 is chosen to strongly encourage exploration given the limited budget (25 steps).
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add exploration term
    # Higher lambda means more exploration
    exploration_weight = 0.5
    scores = ei + exploration_weight * sigma_safe
    
    return scores
# EVOLVE-BLOCK-END
