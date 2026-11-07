"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0  # wall-clock seconds total allowed


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses a combination of Expected Improvement (EI) and UCB-style exploration
    to balance exploitation and exploration for the multi-modal landscape.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB component: prioritize low mean (good prediction) and high uncertainty
    # Normalize mu and sigma to [0, 1] range for balanced combination
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma_safe), np.max(sigma_safe)
    
    mu_range = mu_max - mu_min + 1e-9
    sigma_range = sigma_max - sigma_min + 1e-9
    
    # Lower mu is better, so invert for "higher is better" scoring
    mu_score = 1.0 - (mu - mu_min) / mu_range
    sigma_score = (sigma_safe - sigma_min) / sigma_range
    
    # EI normalization
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_normalized = (ei - ei_min) / ei_range
    
    # Combine: weight EI heavily, add UCB exploration
    # The balance is tuned for Styblinski-Tang's multi-modal nature
    ei_weight = 0.7
    mu_weight = 0.15
    sigma_weight = 0.15
    
    score = ei_weight * ei_normalized + mu_weight * mu_score + sigma_weight * sigma_score
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
