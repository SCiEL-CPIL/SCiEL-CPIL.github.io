"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    Blends Expected Improvement (EI) and Probability of Improvement (PI)
    with an uncertainty bonus, then divides by cost.
    """
    # Safety: ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Standardize for numerical stability in z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Uncertainty bonus: encourages exploration in high-uncertainty regions
    # Normalize sigma to [0,1] relative to its range for a bounded bonus
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Blend: 0.7 * EI + 0.2 * PI + 0.1 * uncertainty bonus
    # EI is in objective units, PI is [0,1], sigma_norm is [0,1]
    # Scale PI and uncertainty to be comparable to EI scale
    # Use a small multiplier for PI and uncertainty to keep EI dominant
    score = 0.7 * ei + 0.2 * pi * np.max(ei) + 0.1 * sigma_norm * np.max(ei)
    
    # Divide by cost
    score = score / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
