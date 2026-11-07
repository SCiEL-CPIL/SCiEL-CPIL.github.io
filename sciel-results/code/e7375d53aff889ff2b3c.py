import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 2.0  # 2 second safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in the tails
    z = np.clip(z, -50, 50)
    
    # Expected Improvement for minimization
    # E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check deadline - EI computation is very fast, this is just a safety net
    if time.time() > deadline:
        return ei
    
    return ei
