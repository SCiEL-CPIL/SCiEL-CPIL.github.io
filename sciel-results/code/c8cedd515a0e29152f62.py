import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Time guard
    if time.time() - _start_time > 320:
        return -mu

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization
    # We want to find x that minimizes f.
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Note: If we use xi, the formula is:
    # EI(x) = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - xi - mu) / sigma
    
    # Let's use a small xi to balance exploration.
    # A common heuristic is xi = 0.01 * abs(f_best) or a small constant.
    # Given the scale is around -1.8, let's try xi = 0.01.
    xi = 0.01
    
    z = (f_best - xi - mu) / sigma
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    ei[~np.isfinite(ei)] = 0.0
    
    return ei
