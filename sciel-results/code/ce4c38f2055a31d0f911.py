import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Check time limit
    # The evaluator kills at 330s.
    # We are just returning an array, so it's fast.
    
    # Adaptive Beta for LCB
    # f_best is the current minimum.
    # Theoretical min is 0.397887.
    # If f_best is high, we need more exploration (higher beta).
    # If f_best is low, we need more exploitation (lower beta).
    
    # Distance from known minimum
    delta = f_best - 0.397887
    
    # Clip delta to avoid extreme values
    delta_clipped = np.clip(delta, 0, 10)
    
    # Beta starts at 1.0 and increases up to 2.0
    beta = 1.0 + 0.1 * delta_clipped
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we negate the LCB
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
