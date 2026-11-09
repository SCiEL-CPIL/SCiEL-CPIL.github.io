"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement (EI) with a small exploration bonus,
    all normalized by cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus to encourage exploration
    # This helps escape local optima while still favoring exploitation
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine EI and exploration bonus
    combined_score = ei + exploration_bonus
    
    # Normalize by cost (cost is at least 1.0, so no division by zero)
    cost_safe = np.maximum(cost, 1e-8)
    scores = combined_score / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
