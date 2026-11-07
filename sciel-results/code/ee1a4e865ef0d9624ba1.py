import numpy as np
from scipy.stats import norm
import time

# Global start time for the guard
_start_time = time.time()
_LIMIT = 320.0  # Safety margin

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _start_time > _LIMIT:
        # Return a safe default, e.g., uniform random or just zeros
        # But we must return 2000 finite scores.
        return np.zeros_like(mu)
    
    z = (f_best - mu) / (sigma + 1e-9)
    return norm.cdf(z)
