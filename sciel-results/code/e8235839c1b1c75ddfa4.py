"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with
    adaptive cost-aware weighting.
    """
    t_start = time.time()
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost normalization: cost is in [1, 2]
    # Use a softer cost penalty that still prefers cheaper points
    # but doesn't be as aggressive as quadratic
    cost_safe = np.maximum(cost, 1e-10)
    
    # Normalize cost to [0, 1] range: cost=1 -> 0, cost=2 -> 1
    cost_norm = (cost_safe - 1.0) / 1.0
    cost_norm = np.clip(cost_norm, 0, 1)
    
    # Cost factor: reduces score for expensive points
    # Use exponential decay for smooth penalty
    cost_factor = np.exp(-1.5 * cost_norm)
    
    # Combine EI and PI with adaptive weighting
    # When EI is high relative to uncertainty, trust EI more
    ei_scale = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ei_norm = ei / ei_scale
    
    # Weight: balance exploration (PI) and exploitation (EI)
    # Higher weight on EI when we have confident predictions
    uncertainty = sigma_safe / (np.mean(sigma_safe) + 1e-10)
    uncertainty = np.clip(uncertainty / 2.0, 0, 1)
    
    w_ei = 0.6 + 0.3 * (1 - uncertainty)  # 0.6 to 0.9
    w_pi = 1.0 - w_ei
    
    # Combined score before cost adjustment
    combined = w_ei * ei_norm + w_pi * pi
    
    # Apply cost-aware penalty
    score = combined * cost_factor
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard check
    elapsed = time.time() - t_start
    if elapsed > 280:
        # Fallback: simple EI with cost penalty
        fallback = ei * cost_factor
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    return score
# EVOLVE-BLOCK-END
