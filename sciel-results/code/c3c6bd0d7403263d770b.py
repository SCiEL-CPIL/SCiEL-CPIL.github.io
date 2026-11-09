import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Time guard: If we are close to the limit, return a simple score to avoid timeout
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return -mu

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-10) # Small epsilon to prevent division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Avoid division by zero issues, though sigma is clamped above
    z = (f_best - mu) / sigma
    
    # Clip z to a reasonable range to avoid numerical issues in cdf/pdf
    # Norm.cdf handles large values well, but let's be safe
    z = np.clip(z, -50, 50)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is non-negative. Higher is better.
    return ei
