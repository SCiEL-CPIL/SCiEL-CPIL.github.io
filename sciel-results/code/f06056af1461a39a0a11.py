# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    n = len(mu)
    
    # Deadline guard
    start_time = time.time()
    deadline = 280.0
    
    # Clamp sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x).
    # A point is promising if it has a low mean (mu) and high uncertainty (sigma).
    # Score = mu - beta * sigma
    # Higher score is better (more negative mu, or larger sigma).
    
    # Beta controls the exploration-exploitation trade-off.
    # beta = 0 -> Greedy (minimize mu)
    # beta -> inf -> Pure exploration (maximize sigma)
    # Typical values are between 1 and 5.
    beta = 2.0
    
    scores = mu - beta * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return -mu + sigma_safe * 0.0
        
    return scores
# EVOLVE-BLOCK-END
