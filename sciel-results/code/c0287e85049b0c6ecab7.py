"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement per unit cost,
    with a small exploration bonus to encourage diversity.
    
    Parameters:
        mu: (2000,) GP mean in objective units
        sigma: (2000,) GP std in objective units
        f_best: lowest observed value
        cost: (2000,) evaluation cost for each candidate
    
    Returns:
        (2000,) acquisition scores (higher = evaluate next)
    """
    # Guard against zero/negative costs
    cost_safe = np.maximum(cost, 1e-12)
    
    # Standardize sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[f_best - f(x)]_+
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    ea = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima and explore new regions
    # The bonus is scaled by 1/sqrt(cost) to be cost-aware
    exploration_bonus = 0.05 * sigma_safe / np.sqrt(cost_safe)
    
    # Final score
    scores = ea + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
