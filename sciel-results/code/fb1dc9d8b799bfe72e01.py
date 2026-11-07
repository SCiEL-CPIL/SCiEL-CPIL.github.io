"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Expected Improvement (EI) normalized by cost, with an exploration bonus.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    # For minimization, EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical issues can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, 1e-10)
    
    # Add exploration bonus to encourage evaluating high-uncertainty, low-cost points
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    scores = ei / cost_safe + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
