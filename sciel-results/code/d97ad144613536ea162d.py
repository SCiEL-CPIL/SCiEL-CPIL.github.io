"""Cost-aware acquisition with diversity promotion for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Diversity-aware, cost-normalized acquisition function.
    
    Combines:
      - Expected Improvement (exploitation)
      - Predictive standard deviation (exploration / information gain)
      - Spatial diversity penalty (kernel density over candidates)
    All normalized by evaluation cost.
    """
    n = len(mu)
    eps = 1e-9
    
    # --- Expected Improvement (standard) ---
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Exploration term: predictive std ---
    # Use sigma directly as a measure of uncertainty
    sigma_safe = np.maximum(sigma, eps)
    
    # --- Spatial diversity penalty ---
    # We don't have coordinates, but we can use the structure of mu/sigma
    # as a proxy: regions where the GP is uncertain AND the mean is far
    # from f_best are likely under-explored.
    # As a diversity signal, we penalize candidates that have very low
    # sigma (already well-predicted) AND are not improving.
    # A cleaner proxy: use the rank of sigma to reward high-variance points.
    
    # Compute a normalized diversity score from sigma
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min + eps
    sigma_norm = (sigma_safe - sigma_min) / sigma_range  # 0 to 1
    
    # Combine EI and exploration
    # Weight: more exploration when EI is small (plateau), less when EI is large
    ei_max = np.max(ei) + eps
    ei_norm = ei / ei_max  # 0 to 1
    
    # Adaptive mixing: when EI is low everywhere, rely more on sigma
    # when EI is high, rely more on EI
    ei_mean = np.mean(ei)
    # If mean EI is very small relative to f_best, we're stuck -> explore more
    stuck_factor = 1.0 / (1.0 + ei_mean / (abs(f_best) + eps))
    # stuck_factor is in (0, 1]: high when stuck
    
    # Blend: alpha controls exploration vs exploitation
    alpha = 0.3 + 0.5 * stuck_factor  # 0.3 to 0.8
    combined = alpha * (sigma_norm * (f_best + eps)) + (1.0 - alpha) * ei
    
    # --- Cost normalization ---
    cost_safe = np.maximum(cost, eps)
    score = combined / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
