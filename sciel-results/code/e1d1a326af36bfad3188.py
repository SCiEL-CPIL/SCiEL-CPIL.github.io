"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) for minimization.
    
    EI(x) = E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu(x), sigma(x)^2)
    
    For minimization, improvement is defined as f_best - f(x).
    The closed form is:
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma, Phi is CDF, phi is PDF of standard normal.
    """
    # Handle numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # z-score for the improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
