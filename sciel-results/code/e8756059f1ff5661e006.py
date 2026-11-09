"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with adaptive exploration.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Basic safety checks
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros_like(mu)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # We want to maximize EI.
    
    # Handle cases where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -100, 100)
    
    # Compute Phi(z) and phi(z)
    # Using standard normal CDF and PDF
    from scipy.stats import norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure finite outputs
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
