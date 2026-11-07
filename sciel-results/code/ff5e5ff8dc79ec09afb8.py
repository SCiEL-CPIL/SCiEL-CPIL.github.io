"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Robust acquisition function combining Expected Improvement with exploration.
    
    Uses a hybrid approach:
    - EI for exploitation
    - Small UCB-like bonus for exploration in uncertain regions
    - Proper numerical handling for small sigma values
    """
    # Ensure sigma is positive and handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # Scale the exploration bonus to be small relative to typical EI values
    # Use a fixed small coefficient to avoid over-exploration
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine: primarily EI with a small exploration component
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
