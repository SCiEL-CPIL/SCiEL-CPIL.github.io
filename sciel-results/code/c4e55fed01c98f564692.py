import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def acquisition(mu, sigma, f_best, cost):
    start = _get_start_time()
    deadline = start + 290.0  # Safety margin for 300s limit, assuming this is called many times? 
    # Actually, the limit is for the *whole* run. 
    # If we assume the run takes ~20s, we have plenty of time.
    # But if we are slow, we should bail.
    
    # The directive says "return the best result found with a safety margin".
    # This implies internal search. We don't have internal search.
    # So we just compute.
    
    # Calculate EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost aware
    # Try penalizing cost more strongly
    scores = ei / (cost * cost)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If time is running out (unlikely for this op, but good practice per directive)
    # We can't really "return best result found" if we haven't started search.
    # So we just return.
    
    return scores
