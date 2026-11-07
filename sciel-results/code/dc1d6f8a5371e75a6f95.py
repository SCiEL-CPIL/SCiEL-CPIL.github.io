"""Acquisition function for Bayesian optimization of beale 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Uses a combination of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    start_time = time.time()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) - higher beta means more exploration
    # For minimization, we use: -mu + beta * sigma
    # We want to maximize this, so we negate mu
    beta = 2.0  # Exploration-exploitation trade-off
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB
    # Normalize both to have similar scales
    ei_max = np.max(ei)
    ucb_max = np.max(ucb)
    
    # If all values are the same, avoid division by zero
    ei_range = ei_max - np.min(ei) if ei_max > np.min(ei) else 1.0
    ucb_range = ucb_max - np.min(ucb) if ucb_max > np.min(ucb) else 1.0
    
    ei_normalized = ei / (ei_range + 1e-12)
    ucb_normalized = ucb / (ucb_range + 1e-12)
    
    # Weighted combination: 70% EI, 30% UCB
    scores = 0.7 * ei_normalized + 0.3 * ucb_normalized
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard: if we've spent too much time, return immediately
    if time.time() - start_time > 0.5:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
