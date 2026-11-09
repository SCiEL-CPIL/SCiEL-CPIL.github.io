"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    Args:
        mu: array of shape (n,) - posterior means
        sigma: array of shape (n,) - posterior standard deviations
        f_best: float - best observed value so far
    
    Returns:
        array of shape (n,) - acquisition scores (higher is better for selection)
    """
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Calculate Z-score for Expected Improvement
    # We are minimizing, so improvement is f_best - f(x)
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF of standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
