"""Rewritten program using Thompson Sampling for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 320.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    mu: array of shape (2000,) - posterior mean in objective units
    sigma: array of shape (2000,) - posterior standard deviation in objective units
    f_best: float - best (lowest) observed value so far
    
    Returns: array of shape (2000,) - scores, higher means more promising
    """
    # Check deadline - if we're close to timeout, return a simple exploitation score
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        # Fallback: pure exploitation (negative mean)
        return -mu
    
    # Thompson Sampling: draw a sample from the posterior predictive distribution
    # For minimization, we want to find where the function is likely to be low
    # Sample: y = mu + sigma * z where z ~ N(0,1)
    # Higher score = more likely to be the minimum
    # So we want to minimize the sample, meaning score = -sample
    
    # Generate a single random draw
    z = np.random.standard_normal(len(mu))
    sample = mu + sigma * z
    
    # Score: lower sample value is better, so negate for "higher is better" scoring
    scores = -sample
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
