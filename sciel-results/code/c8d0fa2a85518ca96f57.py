"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Parameters:
    - mu: array of shape (N,) with posterior mean
    - sigma: array of shape (N,) with posterior standard deviation
    - f_best: scalar, the lowest observed value so far
    
    Returns:
    - scores: array of shape (N,) with acquisition scores (higher is better for exploration)
    """
    # Clip sigma to avoid division by zero and ensure positivity
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Beta parameter for UCB - balances exploration vs exploitation
    # A larger beta encourages more exploration
    beta = 1.0
    
    # UCB for minimization: we want to maximize -mu + beta * sigma
    # Equivalently, score = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
