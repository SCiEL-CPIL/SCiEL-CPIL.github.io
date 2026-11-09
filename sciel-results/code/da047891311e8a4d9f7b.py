"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: Expected Improvement with exploration bonus.
    
    Args:
        mu: Array of means (2000,)
        sigma: Array of standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,). Higher score indicates the candidate 
        is more likely to be the next evaluation point.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus proportional to sigma
    # This helps in early stages when sigma is large
    exploration_bonus = 0.1 * sigma
    
    # Combined score: higher is better
    scores = ei + exploration_bonus
    
    return scores
# EVOLVE-BLOCK-END
