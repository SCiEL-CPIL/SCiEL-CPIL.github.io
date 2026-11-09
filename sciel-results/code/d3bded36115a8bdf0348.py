import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin check (optional, but good practice)
    # elapsed = time.time() - _start_time
    # if elapsed > 320: return np.zeros(len(mu))

    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()
    
    # Ensure numerical stability
    sigma = np.clip(sigma, 1e-12, None)
    cost = np.clip(cost, 1e-12, None)
    
    # Structural Change: Use Upper Confidence Bound (UCB) instead of Expected Improvement (EI)
    # For minimization, we want to minimize mu - beta * sigma
    # We maximize the score: (beta * sigma - mu) / cost
    # This encourages exploring regions with high uncertainty (sigma) and low mean (mu),
    # normalized by the evaluation cost.
    
    beta = 1.0
    
    score = (beta * sigma - mu) / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
