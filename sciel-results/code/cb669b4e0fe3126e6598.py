"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 300.0  # seconds, with safety margin

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using adaptive Upper Confidence Bound (UCB).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        The best (lowest) objective value observed so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher is better.
    """
    # Time guard: if we're close to the limit, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5.0:
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)
        mu = np.where(np.isfinite(mu), mu, f_best)
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle numerical issues
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Adaptive UCB for minimization: maximize -mu + beta * sigma
    # Use adaptive beta based on uncertainty spread and progress
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # If sigma is very small everywhere, we're in exploitation mode
    # If sigma has high variance, we should explore more
    if sigma_mean > 0 and np.isfinite(sigma_mean):
        # Normalize by typical scale
        scale = sigma_mean + 1e-10
        # Beta starts high for exploration, decreases as we find good points
        # f_best closer to 0 means we're doing well, so explore less
        progress = 1.0 / (1.0 + abs(f_best) * 5.0)
        # Base beta with exploration bonus when uncertainty is high
        base_beta = 1.5 + 2.0 * (1.0 - progress)
        # Additional exploration if there's high relative uncertainty
        rel_uncertainty = (sigma_max - sigma_mean) / (sigma_max + 1e-10)
        beta = base_beta * (1.0 + 0.5 * rel_uncertainty)
    else:
        beta = 2.0
    
    # Cap beta to avoid excessive exploration
    beta = np.clip(beta, 0.5, 4.0)
    
    scores = -mu + beta * sigma
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
