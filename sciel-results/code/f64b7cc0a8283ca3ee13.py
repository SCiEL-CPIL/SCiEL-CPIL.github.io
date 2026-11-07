"""Rewritten acquisition for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Variance-gated Expected Improvement acquisition function.
    
    Scales EI by (sigma / sigma_max)^gamma to prioritize exploration
    in high-uncertainty regions, helping escape local minima in Branin.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and ensure positivity
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Variance gating: scale by (sigma / sigma_max)^gamma
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        gamma = 1.5
        variance_gate = (sigma_safe / sigma_max) ** gamma
    else:
        variance_gate = np.ones_like(sigma_safe)
    
    # Combine: use a blend of pure EI and variance-gated EI
    # This maintains exploitation capability while boosting exploration
    alpha = 0.5
    score = alpha * ei + (1 - alpha) * ei * variance_gate
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
