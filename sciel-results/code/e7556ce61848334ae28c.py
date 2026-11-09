"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: return early if we're getting close to the 330s wall-clock limit
_start_time = time.time()
_DEADLINE_MARGIN = 15.0  # seconds of safety margin
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using Thompson Sampling.
    
    This approach samples a value from the posterior distribution at each
    candidate point. The candidate with the lowest sampled value is selected
    (highest score). This provides a natural balance between exploration
    and exploitation, often outperforming deterministic EI/UCB hybrids in
    multimodal landscapes by escaping local minima.
    
    Parameters:
    mu: array-like, shape (N,), mean of the posterior distribution
    sigma: array-like, shape (N,), standard deviation of the posterior distribution
    f_best: float, the best (lowest) observed function value so far
    
    Returns:
    scores: numpy array, shape (N,), higher values indicate better candidates
    """
    global _call_count
    
    # Check deadline
    if time.time() - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Fallback to simple UCB if time is running out
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
        sigma = np.maximum(sigma, 1e-9)
        scores = -mu + 2.0 * sigma
        return np.where(np.isfinite(scores), scores, 0.0)
    
    _call_count += 1
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Draw a sample from N(mu, sigma) for each candidate
    # We negate the sample because we want higher scores to indicate better candidates
    # (lower function values)
    samples = np.random.normal(loc=mu, scale=sigma)
    
    # Ensure all scores are finite
    scores = -samples
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
