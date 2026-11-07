# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses a hybrid of Expected Improvement (EI) and an explicit exploration bonus
    (proportional to standard deviation) to better handle multimodal landscapes
    with a small evaluation budget.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourages visiting regions with high uncertainty
    # A weight of 1.0 is chosen to balance exploitation (EI) and exploration (sigma)
    # effectively for the small budget (25 evaluations).
    exploration_bonus = 1.0 * sigma_safe
    
    # Combined score
    score = ei + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
