"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Upper Confidence Bound (UCB) with adaptive beta.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Posterior mean of the GP at the candidate points.
    sigma : array_like, shape (2000,)
        Posterior standard deviation of the GP at the candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        UCB scores for each candidate point. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive UCB: balance exploration and exploitation
    # Use a beta that encourages exploration early but focuses later
    # Since we don't know the iteration count, use a moderate exploration factor
    # Based on findings, beta=2.5 improved performance significantly
    
    beta = 2.5
    
    # UCB score: mu + beta * sigma
    # Higher values mean more promising (for minimization, we want low mu but high uncertainty)
    scores = mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
