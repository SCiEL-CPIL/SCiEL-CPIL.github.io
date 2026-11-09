"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        The lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        EI scores; higher means evaluate next.
    """
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    from scipy.stats import norm
    
    # Handle sigma near zero to avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Standardized distance from f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
