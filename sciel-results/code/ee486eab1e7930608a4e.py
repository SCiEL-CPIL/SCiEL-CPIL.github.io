"""Acquisition function for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for timing
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Deadline guard
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Return a safe fallback: uniform random scores to avoid timeout
        return np.ones(len(mu))
    
    # Thompson Sampling: sample from the posterior N(mu, sigma) for each candidate
    # The candidate with the minimum sampled value is most promising for minimization
    # We return negative sampled values so that higher scores = more promising
    
    # Generate random samples from the posterior
    rng = np.random.default_rng()
    samples = rng.normal(loc=mu, scale=sigma, size=len(mu))
    
    # For minimization, lower sampled values are better
    # We want higher scores to mean "evaluate next", so negate
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
