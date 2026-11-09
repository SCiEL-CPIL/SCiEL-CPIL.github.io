import numpy as np
from scipy.stats import norm
import time

_t0 = time.time()
_DEADLINE = 295.0  # safety margin before 300s

def acquisition(mu, sigma, f_best, cost):
    elapsed = time.time() - _t0
    if elapsed > _DEADLINE:
        return np.zeros_like(mu)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # EI for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-normalized EI
    score = ei / np.maximum(cost, 1e-9)
    
    # Add small exploration bonus
    score += 0.1 * sigma / np.maximum(cost, 1e-9)
    
    return score
