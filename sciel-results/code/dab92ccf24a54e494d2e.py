import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)

    sigma = np.maximum(sigma, 1e-6)
    
    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is norm.cdf, phi is norm.pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware normalization
    # We want to maximize EI per unit cost
    scores = ei / (cost + 1e-9)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
