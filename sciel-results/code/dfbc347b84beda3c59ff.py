# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    EI balances exploration and exploitation by maximizing the expected amount
    of improvement over the current best value f_best.
    
    For a Gaussian posterior N(mu, sigma^2), the EI is:
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma.
    
    We handle the case where sigma is very small to avoid numerical issues.
    """
    # Ensure sigma is positive and handle near-zero values
    # If sigma is very small, the distribution is effectively a point mass.
    # If mu < f_best, improvement is f_best - mu. Otherwise 0.
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / safe_sigma
    
    # Calculate EI
    # norm.cdf is Phi, norm.pdf is phi
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
