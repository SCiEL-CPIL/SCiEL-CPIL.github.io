import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Start time to track deadline (though this function is fast)
    start_time = time.time()
    
    # Upper Confidence Bound for minimization
    # Score = -mu + beta * sigma
    # Higher score is better
    
    beta = 2.0
    score = -mu + beta * sigma
    
    # Safety margin: if we are taking too long (unlikely), return early?
    # Since it's vectorized, it won't take long.
    
    return score
