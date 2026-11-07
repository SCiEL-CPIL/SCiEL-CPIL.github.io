"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute an acquisition score for each candidate point.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising points.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement (EI) acquisition function
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    improvement = f_best - mu
    z = improvement / sigma
    
    # Handle numerical stability for large |z|
    # For very negative z (large improvement), EI ≈ improvement
    # For very positive z (negative improvement), EI ≈ 0
    
    # Use scipy's norm.cdf and norm.pdf for accuracy
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Compute EI
    ei = improvement * cdf + sigma * pdf
    
    # Ensure non-negative scores (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Return finite scores
    return ei
# EVOLVE-BLOCK-END
