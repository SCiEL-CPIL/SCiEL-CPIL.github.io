import numpy as np
import time

_start_time = None
_DEADLINE = 325.0 # 5 seconds margin from 330s

def acquisition(mu, sigma, f_best):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    
    # Deadline guard
    if elapsed > _DEADLINE:
        # Return a simple, fast score. 
        # If we are this close to the limit, we just want to finish.
        # Pure exploitation is a safe fallback.
        return -mu
    
    # UCB Acquisition Function for Minimization
    # We want to maximize the score.
    # Score = -mu + beta * sigma
    # mu is the mean prediction (lower is better)
    # sigma is the standard deviation (higher means more uncertainty)
    
    beta = 1.5
    
    # Calculate scores
    # Ensure sigma is not zero to avoid division issues if we were doing z-scores, 
    # but for UCB we just add.
    scores = -mu + beta * sigma
    
    # Ensure finiteness
    # If mu or sigma are non-finite, the result might be non-finite.
    # The prompt guarantees mu and sigma are valid, but "Return 2000 finite scores" is a strict constraint.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
