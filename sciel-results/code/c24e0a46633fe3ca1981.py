# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI).
    """
    start_time = time.time()
    deadline = 330.0  # Seconds, with margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate Probability of Improvement (PI)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf for extreme values
    z = np.clip(z, -10, 10)
    
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Ensure finite
    score = np.where(np.isfinite(pi), pi, 0.0)
    
    # Check deadline (unlikely to be hit for 2000 points)
    if time.time() - start_time > deadline:
         return score
         
    return score
# EVOLVE-BLOCK-END
