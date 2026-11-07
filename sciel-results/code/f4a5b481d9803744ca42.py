import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    # If we are getting close to the limit, return a simple, fast score
    # to ensure we don't hang or crash due to timeout in the harness.
    # However, since this is vectorized, it's very fast.
    # The guard is mostly for compliance with the directive.
    if time.time() - _start_time > 290:
        # Fallback: simple random-ish or just zeros to be safe?
        # Returning zeros means the evaluator picks the first one?
        # Better to return a valid score.
        return np.ones(len(mu))

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # Cost-aware score: PI / cost
    # cost is shape (2000,)
    # Avoid division by zero (cost >= 1 anyway)
    scores = pi / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
