import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Time guard: return safe zeros if we are close to the 330s limit
    if time.time() - _start_time > 325:
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive for stable division
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate components for EI
    # phi(z) is the standard normal PDF
    # Phi(z) is the standard normal CDF
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # EI should be non-negative, clip small negative values due to numerical error
    ei = np.maximum(ei, 0.0)
    
    # Ensure all values are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
