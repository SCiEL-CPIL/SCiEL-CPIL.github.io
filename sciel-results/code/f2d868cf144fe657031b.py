"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_bo_start_time = None
_bo_call_count = 0

def acquisition(mu, sigma, f_best):
    global _bo_start_time, _bo_call_count
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
    
    _bo_call_count += 1
    
    # Deadline Guard: 330s limit, use 305s as safety margin
    if time.time() - _bo_start_time > 305:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        # Fallback to pure exploitation (minimize mu)
        return -np.asarray(mu)

    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Structural Change: Marginal Thompson Sampling
    # Instead of deterministic Expected Improvement (EI), we use stochastic
    # Thompson Sampling. We draw a sample from the posterior predictive
    # distribution at each candidate point. Since we only have marginal
    # statistics (mu, sigma), we sample from N(mu, sigma^2) independently
    # for each point.
    
    # Clip sigma to avoid division by zero or numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Generate random normal samples
    # np.random.randn returns standard normal samples
    z = np.random.randn(len(mu))
    
    # Draw samples from the posterior: f ~ N(mu, sigma^2)
    samples = mu + sigma_safe * z
    
    # We want to minimize the objective function.
    # Thompson Sampling suggests evaluating the point with the lowest sample.
    # The acquisition function should return scores where HIGHER is better.
    # Therefore, we return the negative of the samples.
    scores = -samples
    
    # Ensure finite values
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
