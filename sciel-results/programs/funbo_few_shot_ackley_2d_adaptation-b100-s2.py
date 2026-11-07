"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_bo_start_time = None

def acquisition(mu, sigma, f_best):
    global _bo_start_time
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
        
    # Deadline Guard: 300s limit, use 280s as safety margin
    if time.time() - _bo_start_time > 280:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        return -mu

    # Structural Change: Upper Confidence Bound (UCB)
    # UCB encourages exploration of high-uncertainty regions more linearly 
    # than Expected Improvement (EI), which can help escape local optima 
    # in multimodal functions like Ackley.
    
    # Adaptive beta:
    # If f_best is high, we are far from the optimum, so explore more (higher beta).
    # If f_best is low, we are close, so exploit more (lower beta).
    # Ackley 2D max value is ~30.
    # Let's scale beta from 0.5 to 3.0 based on f_best.
    # If f_best > 10, beta ~ 3.0
    # If f_best < 0.1, beta ~ 0.5
    
    # Simple heuristic:
    beta = 0.5 + 2.5 * min(1.0, f_best / 10.0)
    
    # Score = -mu + beta * sigma
    # We want to minimize f, so we want low mu.
    # We want to explore, so we want high sigma.
    # Higher score is better.
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
