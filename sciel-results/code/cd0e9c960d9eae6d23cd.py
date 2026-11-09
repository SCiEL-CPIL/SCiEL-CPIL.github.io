"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blends Expected Improvement with a UCB-like exploration term.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip extreme values for numerical stability
    ei = np.clip(ei, -1e6, 1e6)
    
    # UCB term: encourages exploration in high-uncertainty regions
    # Use a moderate exploration coefficient
    beta = 1.5
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want low mu and high sigma
    # EI already balances exploitation (low mu) and exploration (high sigma)
    # But in flat regions, EI can be weak; add a small UCB-like boost
    
    # Blend: primarily EI, with a small UCB component to break plateaus
    # The UCB term here is used as a bonus for high uncertainty
    # We normalize both to comparable scales
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # For minimization, lower UCB is better, so invert
    ucb_norm_inv = 1.0 - ucb_norm
    
    # Blend with 80% EI and 20% inverted UCB
    scores = 0.8 * ei_norm + 0.2 * ucb_norm_inv
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
