"""Rewritten program using a novel acquisition strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses Expected Improvement (EI) to balance exploitation and exploration.
    EI calculates the expected improvement over the current best value,
    which naturally adapts the exploration/exploitation trade-off based on
    the posterior distribution.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute PDF and CDF of the standard normal distribution
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    scores = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure all scores are finite and non-negative
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
