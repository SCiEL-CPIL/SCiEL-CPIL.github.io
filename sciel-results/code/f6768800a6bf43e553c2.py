"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
# Safety margin: The limit is 330s. We aim to finish within 320s to be safe.
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best):
    # Explicit deadline guard
    if time.time() > _DEADLINE:
        # Fallback: Greedy selection based on mean
        # Return -mu so that the max score corresponds to the min mu
        return -np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Ensure sigma is non-negative and finite
    sigma = np.abs(sigma)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e-6, neginf=1e-6)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    
    # Add small epsilon to sigma to prevent division by zero
    eps = 1e-8
    sigma_safe = sigma + eps
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate cdf and pdf
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate EI
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure finite values
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
