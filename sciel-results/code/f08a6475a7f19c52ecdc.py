import numpy as np
import time

# Global timer for deadline guard
_START_TIME = time.time()
_LIMIT = 300.0

def acquisition(mu, sigma, f_best):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _LIMIT - 5.0:
        # Fallback to greedy: pick lowest mu
        # Return scores such that lowest mu has highest score
        # Score = -mu
        return -mu

    # UCB for minimization: Maximize -mu + beta * sigma
    # beta = 2.0
    beta = 2.0
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    # If sigma is 0, score is -mu.
    # If mu or sigma is inf/nan, handle it.
    # The task says "Return 2000 finite scores".
    # We should replace non-finite values with a very low score (so they aren't picked)
    # or very high? 
    # "higher meaning evaluate next".
    # So invalid points should have very low scores.
    
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
