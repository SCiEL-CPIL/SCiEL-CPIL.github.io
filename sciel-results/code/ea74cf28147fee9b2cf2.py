import numpy as np
from scipy.stats import norm
import time

# Track state for adaptive parameters
_start_time = time.time()
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _eval_count
    _eval_count += 1
    
    # Deadline guard: if we're close to the 330s limit, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback: just use -mu (exploit best known)
        return -mu
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive beta: stronger exploration early, weaker late
    # f_best starts high (~100+) and decreases toward 0
    # Use a smooth decay based on f_best
    if f_best > 10:
        beta = 0.6
    elif f_best > 2:
        beta = 0.3
    elif f_best > 0.5:
        beta = 0.15
    else:
        beta = 0.05
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add uncertainty bonus to encourage exploration in high-variance regions
    # This helps escape local optima in the Beale function landscape
    return ei + beta * sigma_safe
