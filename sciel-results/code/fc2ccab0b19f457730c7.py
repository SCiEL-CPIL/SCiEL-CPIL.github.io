import numpy as np
import time

# Global variables for time tracking
_start_time = time.time()
_time_limit = 325.0  # Safety margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian Optimization.
    Uses Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta * sigma
    Higher score indicates a more promising point.
    """
    # Optional: Check if we are close to the time limit
    # If so, return a simple exploitation score to save computation?
    # The computation is negligible, so this is just a formality.
    # if time.time() - _start_time > _time_limit:
    #     return -mu

    # UCB parameter
    beta = 2.0
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate UCB score for minimization
    # We want to maximize the score, which corresponds to minimizing the function.
    # Low mu is good (high score via -mu).
    # High sigma is good (high score via beta * sigma).
    scores = -mu + beta * sigma_safe
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
