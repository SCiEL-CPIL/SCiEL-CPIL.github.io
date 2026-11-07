import numpy as np
import time

_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Safety margin: if we've used > 290s, return a simple fallback
    # But we must return valid scores.
    # If we are close to timeout, maybe just return -mu (greedy) to be fast?
    # -mu is very fast.
    
    elapsed = time.time() - _START_TIME
    if elapsed > 290:
        return -mu
    
    beta = 2.0
    lambda_cost = 0.5
    
    scores = -mu + beta * sigma - lambda_cost * cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
