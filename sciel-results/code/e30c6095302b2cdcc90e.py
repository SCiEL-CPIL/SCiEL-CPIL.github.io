"""Rewritten acquisition for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) acquisition score for minimization.
    
    EI(x) = E[max(f_best - f(x), 0)]
    
    We return the negative of the improvement potential so that higher scores
    indicate better candidates to evaluate next.
    """
    # Standardize sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Z-score: how many standard deviations below f_best the mean is
    z = (f_best - mu) / sigma
    
    # Expected improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute Phi(z) and phi(z)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # EI calculation
    ei = (f_best - mu) * Phi + sigma * phi
    
    # For minimization, we want to maximize EI
    # But the interface expects higher scores to mean "evaluate next"
    # So we return EI directly (higher EI = better to evaluate)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Clip to non-negative values (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
