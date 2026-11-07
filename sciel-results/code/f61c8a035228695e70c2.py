import numpy as np
import time
from scipy.stats import norm

_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we've been running too long, return quickly
    elapsed = _get_elapsed()
    
    # Practical Knowledge Gradient / Expected Improvement for minimization
    # KG(x) = E[max(f_best - f(x), 0)]
    # = sigma * (z * Phi(z) + phi(z)) where z = (f_best - mu) / sigma
    
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    
    # Use scipy's norm for numerical stability
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = sigma_safe * (z * Phi_z + phi_z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
