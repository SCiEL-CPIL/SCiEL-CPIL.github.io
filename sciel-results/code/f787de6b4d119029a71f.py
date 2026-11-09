"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 250.0
    
    # Structural Change: Switch from LogEI to UCB (Upper Confidence Bound)
    # UCB provides stronger exploration pressure by explicitly rewarding uncertainty (sigma).
    # For minimization, we maximize the "optimistic lower bound" of the negative function.
    # Score = -mu + beta * sigma
    # This encourages exploring regions with high uncertainty (high sigma) and regions with low mean (low mu).
    
    beta = 2.5  # Exploration-exploitation trade-off parameter
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate UCB scores
    scores = -mu + beta * sigma_safe
    
    # Check time
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
