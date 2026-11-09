import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)

    sigma = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # This is a different modeling family from LCB
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: we want to maximize EI per unit cost
    # Use a softer cost penalty than pure division
    cost_penalty = np.sqrt(cost_safe)
    
    scores = ei / cost_penalty
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
