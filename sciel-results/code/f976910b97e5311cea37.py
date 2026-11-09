import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 10 seconds
    elapsed = time.time() - _start_time
    if 330 - elapsed < 10:
        # Fallback: Return scores based on mean (exploitation)
        # Ensure finite
        return np.where(np.isfinite(mu), -mu, 0.0)

    # Structural Change: UCB (Upper Confidence Bound) for minimization
    # Score = -mu + beta * sigma
    # Higher score means more likely to be the minimum (low mu) or high uncertainty (high sigma)
    beta = 1.5
    sigma_safe = np.maximum(sigma, 1e-9)
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
