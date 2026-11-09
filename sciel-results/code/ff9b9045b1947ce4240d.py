"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > 300:
        return np.zeros_like(mu)
    
    # Expected Improvement for minimization
    # Improvement = f_best - f(x)
    # E[I] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Handle sigma == 0 safely
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We want to maximize EI, so higher score is better
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
