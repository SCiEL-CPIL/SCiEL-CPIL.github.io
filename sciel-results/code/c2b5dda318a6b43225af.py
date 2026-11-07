"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses Thompson Sampling (Posterior Sampling) which provides a robust
balance between exploration and exploitation by sampling from the
GP posterior and selecting the point with the lowest sampled value.
This is a structural change from Expected Improvement (deterministic
expectation) to stochastic sampling, which can escape local optima
more effectively in low-dimensional spaces.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using Thompson Sampling.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far (unused in TS, but kept for interface)
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline safety check
    if time.time() > _DEADLINE - 5.0:
        # Fallback: UCB-like score
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    # Ensure sigma is strictly positive
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling:
    # 1. Draw a sample from the posterior distribution at each candidate point.
    #    y_i ~ N(mu_i, sigma_i^2)
    # 2. The score is the negative of this sample, because we want to MINIMIZE y.
    #    Higher score => Lower sampled y => More likely to be the minimum.
    
    # Generate standard normal noise
    # np.random.randn is efficient for vectorized operations
    noise = np.random.randn(len(mu))
    
    # Compute sampled objective values
    y_samples = mu + sigma_safe * noise
    
    # Convert to acquisition scores (higher is better)
    # We want to pick the point with the lowest y_sample.
    scores = -y_samples
    
    # Numerical cleanup: ensure scores are finite
    # If a sample is extremely negative, it's a good candidate.
    # If a sample is extremely positive, it's a bad candidate.
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
# EVOLVE-BLOCK-END
