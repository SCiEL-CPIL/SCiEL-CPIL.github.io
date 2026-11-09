"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
# The total time limit is 330s. We set a safety margin to return a fallback
# if we are dangerously close to the limit, although this function is very fast.
_TIMEOUT_MARGIN = 325.0

def acquisition(mu, sigma, f_best):
    # Explicit deadline guard
    if time.time() - _start_time > _TIMEOUT_MARGIN:
        # Fallback: Return negative mean (greedy/exploitation) to finish quickly
        return -mu

    # Thompson Sampling
    # Sample from the posterior predictive distribution at the candidate points.
    # Since we are minimizing, we want the candidate with the lowest expected value.
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled value.
    
    # Clip sigma to avoid zero or negative values which cause errors in normal sampling
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw samples from the Gaussian posterior
    # This introduces stochasticity that helps escape local optima
    samples = np.random.normal(mu, sigma_safe)
    
    # Higher score means "evaluate next". 
    # Low function value is good. 
    # So -low_value is a high score.
    return -samples
# EVOLVE-BLOCK-END
