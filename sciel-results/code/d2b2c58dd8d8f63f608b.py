"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Use a hybrid acquisition function combining Expected Improvement (EI)
    and a UCB-like exploration term to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher is better.
    """
    # Clip sigma to avoid division by zero or negative values
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) component
    z = (f_best - mu) / sigma
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Combine EI with a raw exploration term (sigma)
    # This preserves the physical scale of the function values.
    # EI dominates when we are close to the optimum (low sigma, high EI potential).
    # Sigma dominates when we are far from the optimum (high sigma, low EI).
    # A weight of 1.0 provides a balanced exploration-exploitation tradeoff.
    exploration_weight = 1.0
    scores = ei + exploration_weight * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
