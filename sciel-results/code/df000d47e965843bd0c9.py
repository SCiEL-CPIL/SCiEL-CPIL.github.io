"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guarding
_start_time = time.time()
TIME_LIMIT = 300.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > TIME_LIMIT:
        # Return a safe fallback: pure exploitation (lower mu is better)
        # Since higher score means evaluate next, we want high scores for low mu.
        # Return -mu as score? Or just zeros?
        # If we return -mu, it's a valid score.
        return -mu

    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # We want to maximize this probability.
    
    # Handle division by zero or very small sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF
    scores = norm.cdf(z)
    
    # Ensure finite scores
    # norm.cdf should always be finite for finite inputs, but just in case
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
