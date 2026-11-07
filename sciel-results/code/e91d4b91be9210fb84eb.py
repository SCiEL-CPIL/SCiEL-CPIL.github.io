"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the Gaussian process posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best observed value so far.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores where higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # UCB for minimization: we want low mu and high sigma.
    # Score = -mu + beta * sigma
    # Higher score is better.
    beta = 3.0
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
