import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Time guard: If we are close to the limit, return a simple score to avoid timeout
    # The calculation is very fast, so this is just a safety net.
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return -mu

    # UCB for minimization: maximize -mu + beta * sigma
    # Based on Inspiration 1 (score 1.1783), beta=2.5 is effective for Hartmann 6D
    beta = 2.5
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
