import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0 # Safety margin

def acquisition(mu, sigma, f_best, cost):
    # Check time
    if time.time() - _start_time > _TIME_LIMIT:
        # Return a safe fallback, e.g., uniform or previous best
        # Since we don't have previous best, return zeros or random
        # Zeros are valid finite scores
        return np.zeros_like(mu)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost-aware UCB
    # We want to minimize f.
    # UCB for minimization: mu - kappa * sigma
    # We want to maximize the "potential improvement" over f_best.
    # Potential = f_best - (mu - kappa * sigma) = f_best - mu + kappa * sigma
    
    kappa = 1.5
    potential = f_best - mu + kappa * sigma
    
    # Normalize by cost
    # Cost is in [1, 2]
    # Using cost ** 0.5 to soften the penalty
    scores = potential / (cost ** 0.5)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
