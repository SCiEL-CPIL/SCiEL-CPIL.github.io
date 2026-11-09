"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global variable to track elapsed time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    
    Draws a sample from the posterior predictive distribution at each candidate:
    y(x) ~ N(mu(x), sigma(x)^2)
    Returns scores where higher score indicates a lower predicted value.
    Score(x) = -y(x) = -mu(x) - sigma(x) * z, where z ~ N(0, 1).
    """
    # Deadline guard: 330s total budget. 
    # This function is called once per step. 
    # If we are close to the limit, we should return quickly.
    # However, TS is O(N) and very fast.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Fallback: return zeros or simple exploitation
        return -mu

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw standard normal samples for each candidate
    z = np.random.standard_normal(len(mu))
    
    # Compute Thompson Sampling scores
    # We want to minimize f(x). 
    # Sample y = mu + sigma * z
    # We select the candidate with the lowest y.
    # The evaluator selects the candidate with the HIGHEST score.
    # So we return -y.
    scores = -mu - sigma_safe * z
    
    # Handle any potential NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
