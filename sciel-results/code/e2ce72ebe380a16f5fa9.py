# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 300.0  # Leave margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Uses Expected Improvement (EI) for minimization.
    
    EI naturally balances exploration and exploitation:
    - High when mu is much lower than f_best (exploitation of good regions)
    - High when sigma is large (exploration of uncertain regions)
    - Low when mu is high and sigma is low (avoid known bad regions)
    
    This is particularly well-suited for the Beale function which has
    a narrow global minimum in a wide flat region.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Emergency fallback: pure exploitation
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite and non-negative
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
