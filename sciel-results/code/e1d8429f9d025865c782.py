"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time to ensure we don't exceed the 300s limit (with safety margin)
_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher meaning more promising.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're getting close to the 300s limit, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 280:  # 20s safety margin
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = -mu + 0.5 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Scale EI by its maximum to normalize to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_scaled = ei / ei_max
    else:
        ei_scaled = np.zeros_like(ei)
    
    # Exploration term: favor points with high sigma but reasonable mu
    # Use a soft penalty for very high mu (poor exploitation)
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range > 1e-10:
        mu_norm = (mu - mu_min) / mu_range  # 0 = best, 1 = worst
    else:
        mu_norm = np.zeros_like(mu)
    
    # Exploration score: high sigma, low mu_norm
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-10:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Adaptive exploration weight: stronger early on, weaker as we converge
    # f_best starts around 5 (uniform random) and minimum is 3
    progress = np.clip((5.0 - f_best) / 2.0, 0, 1)  # 0 at start, 1 when f_best=3
    # Exploration weight decreases from 0.25 to 0.05 as progress increases
    exp_weight = 0.25 - 0.20 * progress
    
    # Exploration boost: strongest where sigma is high and mu is not terrible
    exploration_boost = exp_weight * sigma_norm * (1.0 - 0.5 * mu_norm)
    
    # Final combined score: primarily EI, with adaptive exploration boost
    scores = ei_scaled + exploration_boost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
