"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (N,)
    sigma (np.ndarray): Posterior standard deviation, shape (N,)
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Scores for each candidate, higher is better.
    """
    # UCB: We want to minimize f, so we prefer low mu.
    # We also want to explore high sigma.
    # Score = -mu + beta * sigma
    
    # Beta controls the exploration-exploitation trade-off.
    # A value around 2.0 is a standard starting point for BO.
    beta = 2.0
    
    # Ensure sigma is non-negative (though it should be from GP)
    sigma = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma
    
    return scores
# EVOLVE-BLOCK-END
