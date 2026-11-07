"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6d.
    
    Uses a Variance-Thresholded Expected Improvement (VT-EI) strategy.
    Standard EI is computed for minimization, then a dynamic variance
    threshold masks out regions where posterior uncertainty (sigma) exceeds
    a multiple of the median sigma, forcing exploitation in well-characterized
    regions while still allowing some exploration.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute standard Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle edge case where ei might be slightly negative due to numerical issues
    ei = np.maximum(ei, 0.0)
    
    # Dynamic variance threshold: 1.5 * median sigma
    median_sigma = np.median(sigma_safe)
    var_threshold = 1.5 * median_sigma
    
    # Mask: only allow EI in regions where sigma is below the threshold
    # For regions above threshold, we use a reduced exploration bonus
    # This creates a soft transition rather than a hard cutoff
    
    # Create a smooth mask: 1 where sigma < threshold, smoothly decreasing above
    # Using a logistic-like transition centered at the threshold
    transition_width = 0.5 * median_sigma + 1e-10
    mask = 1.0 / (1.0 + np.exp((sigma_safe - var_threshold) / transition_width))
    
    # Apply the mask to EI scores
    scores = ei * mask
    
    # Add a small exploration term for high-uncertainty regions to avoid
    # completely ignoring them, but much smaller than the EI term
    exploration_bonus = 0.1 * sigma_safe * (1.0 - mask)
    scores = scores + exploration_bonus
    
    # Check deadline
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
