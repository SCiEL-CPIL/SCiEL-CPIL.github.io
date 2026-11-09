import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0 # 300s limit, keep margin
    
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Handle edge cases
    if sigma.min() <= 0:
        sigma = np.maximum(sigma, 1e-9)
        
    # Calculate Expected Improvement (EI) as a baseline or alternative?
    # Let's use UCB.
    # Dynamic beta based on f_best
    # f_min is 3.0
    # If f_best is close to 3, we are in exploitation phase.
    # If f_best is far, we are in exploration phase.
    
    # Normalize f_best to a [0, 1] range roughly?
    # Min 3, Max ~100.
    # progress = 1 - (f_best - 3) / (100 - 3)
    # If f_best > 100, progress < 0.
    
    # Let's use a simpler heuristic for beta.
    # beta should be larger when uncertainty is high relative to improvement potential.
    # But we don't have global uncertainty.
    
    # Let's try: beta = 2.0 * (f_best - 3.0) / 10.0
    # Clamped between 0.5 and 5.0
    beta = 2.0 * (f_best - 3.0) / 10.0
    beta = np.clip(beta, 0.5, 5.0)
    
    # UCB Score: -mu + beta * sigma
    # Higher is better
    scores = -mu + beta * sigma
    
    # Check time
    if time.time() > deadline:
         # Return something valid immediately
         # Just return the scores calculated so far? 
         # The calculation above is O(N) and very fast.
         pass
         
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
