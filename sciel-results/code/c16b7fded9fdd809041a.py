import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle potential numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement (EI) for minimization
    # Improvement is f_best - f(x)
    # E[max(0, f_best - f(x))]
    
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical instability in norm.cdf and norm.pdf
    # norm.cdf(-10) ~ 0, norm.cdf(10) ~ 1
    # norm.pdf(10) ~ 0
    z = np.clip(z, -10, 10)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return ei
        
    # Return scores
    return ei
