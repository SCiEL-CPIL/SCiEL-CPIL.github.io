"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
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
        return -mu
    
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Structural Change: Thompson Sampling
    # Instead of deterministic EI/UCB, we use Thompson Sampling which samples
    # from the posterior and picks the point with the lowest sampled value.
    # This provides stochastic exploration that adapts naturally to the
    # posterior uncertainty without needing explicit exploration parameters.
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: sample from N(mu, sigma)
    # We add a small noise scaling factor to balance exploration/exploitation
    # Based on findings, scaling by 0.5 reduces variance and improves performance
    noise_scale = 0.5
    samples = mu + noise_scale * sigma_safe * np.random.randn(len(mu))
    
    # For minimization, we want the lowest sampled value
    # So the score should be negative of the sample (higher score = lower value)
    scores = -samples
    
    # Ensure finite values
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
