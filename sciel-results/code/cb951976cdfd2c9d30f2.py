"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining expected improvement and
    expected hypervolume improvement concepts, with cost normalization.
    
    Uses a combination of:
    1. Expected Improvement (EI) - standard BO acquisition
    2. Cost-adjusted term to favor cheaper evaluations
    3. Exploration bonus for high uncertainty regions
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Expected Improvement
    delta = f_best - mu
    z = delta / sigma_safe
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware weighting: divide by cost to prefer cheap evaluations
    # But also add a small exploration term based on sigma
    # The idea: we want high EI per unit cost, but also some exploration
    
    # Primary: EI per cost
    score_primary = ei / cost_safe
    
    # Secondary: exploration bonus for uncertain regions, also cost-adjusted
    # This helps when EI is zero (mu >= f_best) but sigma is high
    exploration_bonus = sigma_safe / (cost_safe * (1.0 + cost_safe))
    
    # Combine: primary dominates, but exploration provides a floor
    # Use a weighted sum where the weight on exploration is small
    score = score_primary + 0.1 * exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
