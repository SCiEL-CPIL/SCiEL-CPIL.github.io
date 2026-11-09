"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Uses Expected Improvement (EI) for minimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the Gaussian process posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        The best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher scores indicate more promising candidates.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    # Phi is the standard normal CDF and phi is the standard normal PDF.
    
    z = (f_best - mu) / sigma
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
