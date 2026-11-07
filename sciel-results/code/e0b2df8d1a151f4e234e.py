import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0 # 10s safety margin
    
    # UCB with adaptive-ish beta
    # For minimization, we want low mu and high sigma.
    # Score = -mu + beta * sigma
    # Beta 1.0 is a standard balance; 2.5 proved too exploratory.
    beta = 1.0
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline (though this is very fast)
    if time.time() > deadline:
         # Return best found (which is just the scores)
         return scores
         
    return scores
