"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function with a small exploration term.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising points.
    """
    # Handle edge cases where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip to avoid numerical issues
    ei = np.maximum(ei, 0.0)
    
    # Return EI scores (higher is better)
    return ei
# EVOLVE-BLOCK-END
