import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
_TIMEOUT = 325.0  # 5 seconds margin before 330s

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Return a deterministic score if timeout is imminent, 
        # though this is unlikely for 2000 points.
        # We just return the argmin of mu (greedy) to be safe.
        return -mu
    
    # UCB for minimization: Score = -mu + beta * sigma
    # We want to maximize this score.
    beta = 1.5
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    # If sigma is 0, scores are -mu.
    # If mu is inf, scores are inf. 
    # The task guarantees mu and sigma are from a GP, so they should be finite.
    # But we can clip or handle nan/inf just in case.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
