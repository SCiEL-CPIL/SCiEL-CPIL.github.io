"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement
    with a small exploration bonus, normalized by evaluation cost.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardized distance from current best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using stable computation
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Small exploration bonus to encourage diversity in high-uncertainty regions
    # This helps avoid premature convergence, especially in multimodal functions
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine EI and exploration, then normalize by cost
    # Cost is in [1, 2] for these problems, so this is a mild penalty
    score = (ei + exploration_bonus) / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
