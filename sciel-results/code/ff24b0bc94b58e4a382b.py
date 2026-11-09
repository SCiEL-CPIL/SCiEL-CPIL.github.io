"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition combining Expected Improvement and Probability of Improvement
    # with adaptive weighting to balance exploration and exploitation.
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Cost normalization: normalize costs to [0, 1] range for better scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-12:
        cost_norm = np.ones_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Invert cost norm so lower cost gets higher weight
    cost_weight = 1.0 / (cost_norm + 0.1)  # Add small epsilon for stability
    
    # Combine EI and PI with adaptive weighting
    # When uncertainty is high, weight PI more; when uncertainty is low, weight EI more
    uncertainty = sigma_safe / (np.max(sigma_safe) + 1e-9)
    pi_weight = 0.3 + 0.4 * uncertainty  # Between 0.3 and 0.7
    ei_weight = 1.0 - pi_weight
    
    # Normalize EI and PI to comparable scales
    ei_max = np.max(ei)
    pi_max = np.max(pi)
    if ei_max < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / ei_max
    if pi_max < 1e-12:
        pi_norm = np.zeros_like(pi)
    else:
        pi_norm = pi / pi_max
    
    # Combined acquisition score
    score = (ei_weight * ei_norm + pi_weight * pi_norm) * cost_weight
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
