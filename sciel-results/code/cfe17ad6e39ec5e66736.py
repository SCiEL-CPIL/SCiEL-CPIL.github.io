"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        # Return a uniform score if timed out, though this shouldn't happen
        # frequently if computation is light. 
        return np.zeros_like(mu)
    
    # LCB formula for minimization: min(mu - beta * sigma)
    # We want to maximize, so we return -mu + beta * sigma
    beta = 3.5
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
