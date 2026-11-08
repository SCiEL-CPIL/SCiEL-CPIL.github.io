import numpy as np
import time

_START_TIME = None

def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if _get_elapsed_time() > 320: # 10s margin
        # Return a safe fallback, e.g., uniform random scores or based on mu only
        # To be safe, return scores that prioritize low mu (exploitation)
        return -mu / cost

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # LCB (Lower Confidence Bound) for minimization
    # We want to maximize (beta * sigma - mu) / cost
    beta = 2.0
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-8)
    
    scores = (beta * sigma - mu) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
