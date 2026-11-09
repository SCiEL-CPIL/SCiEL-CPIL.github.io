"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        EI scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and handle numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected improvement for minimization
    # improvement = f_best - mu (how much better mu is than f_best on average)
    improvement = f_best - mu
    
    # z = improvement / sigma
    z = improvement / sigma
    
    # EI = improvement * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = improvement * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI should be non-negative by definition; clip small negative values to 0
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
