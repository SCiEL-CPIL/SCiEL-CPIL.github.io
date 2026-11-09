"""Rewritten acquisition: Expected Improvement (EI) with a small exploration floor."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute the Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu: array of shape (N,) - mean of the GP posterior at candidate points
    sigma: array of shape (N,) - standard deviation of the GP posterior at candidate points
    f_best: float - the best (lowest) observed function value so far
    
    Returns:
    scores: array of shape (N,) - EI values, higher means more promising
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical issues
    scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    # Add a tiny exploration bonus proportional to sigma to encourage exploration
    # when EI is very small (e.g., near f_best), helping escape local plateaus
    exploration_weight = 0.05
    scores = scores + exploration_weight * sigma_safe
    
    return scores
# EVOLVE-BLOCK-END
