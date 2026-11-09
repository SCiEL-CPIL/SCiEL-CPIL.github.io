"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # Higher score means more likely to be a better (lower) value
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # For very small sigma, EI approaches max(f_best - mu, 0)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score: how many std devs below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # For points where sigma is extremely small and mu < f_best, EI should be close to f_best - mu
    # For points where mu >= f_best, EI should be close to 0
    # The formula above handles this correctly, but let's ensure non-negative scores
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
