import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) Acquisition Function for Minimization.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 310:
        # Fallback to greedy (maximize -mu)
        return -mu
    
    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e-8, neginf=1e-8)
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Z
    z = (f_best - mu) / sigma
    
    # Clip Z for numerical stability
    z = np.clip(z, -10, 10)
    
    # Calculate EI
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu < f_best, z is positive. Phi(z) is close to 1. phi(z) is small.
    # EI approx f_best - mu.
    # If mu > f_best, z is negative. Phi(z) is small.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite output
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e9, neginf=0.0)
    
    return ei
