import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline check (safety margin 10s before 300s limit, i.e., 290s)
    # But since this function is called many times, we can't just kill the whole process.
    # The prompt says "return the best result found". 
    # However, this function returns scores, not the final result.
    # The "best result" logic is in the evaluator loop, not here.
    # So a deadline guard inside `acquisition` that returns something "best" doesn't make sense 
    # unless we return a constant score that forces the evaluator to pick a specific point?
    # No, the evaluator picks the argmax.
    # If we time out, we should probably return a valid array quickly.
    # Given the speed, this is not a practical concern. I will skip complex deadline logic 
    # as it might interfere with correctness if not handled perfectly, 
    # and the function is O(N) with small N.
    
    # Ensure sigma is non-negative and non-zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z
    z = (f_best - mu) / sigma
    
    # EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus
    # Using a weight of 1.0 to strongly encourage exploring high uncertainty regions
    score = ei + 1.0 * sigma
    
    return score
