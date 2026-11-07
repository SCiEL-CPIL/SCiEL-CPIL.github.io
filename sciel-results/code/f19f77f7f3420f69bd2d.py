import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 280.0  # 20s margin
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # UCB Acquisition for Minimization
    # Maximize: -mu + beta * sigma
    
    # Heuristic Beta
    # Using a fixed beta of 2.0 for balanced exploration/exploitation
    beta = 2.0
    
    # Clip sigma to avoid zero division or negative values (though GP sigma should be >= 0)
    sigma = np.clip(sigma, 1e-9, None)
    
    # Calculate scores
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline (unlikely to be hit, but required by directive)
    if time.time() > deadline:
        # If we were doing iterative steps, we'd stop here.
        # For direct formula, we just return.
        pass
        
    return scores
