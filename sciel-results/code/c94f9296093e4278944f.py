"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 5 seconds safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 3d.
    
    Args:
        mu: array of shape (2000,), posterior mean at candidate points
        sigma: array of shape (2000,), posterior standard deviation at candidate points
        f_best: lowest observed value so far
    
    Returns:
        array of shape (2000,) with scores, higher meaning evaluate next
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    # Handle edge cases with numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI) acquisition for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in extreme cases
    z = np.clip(z, -10.0, 10.0)
    
    # Compute EI using stable formulation
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure all values are finite and non-negative
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
