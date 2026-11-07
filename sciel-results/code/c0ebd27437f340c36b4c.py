"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid approach:
    - Expected Improvement (EI) for exploitation
    - Uncertainty bonus for exploration
    - Cost penalty for expensive points
    
    The acquisition score is designed so that higher scores indicate
    points that are more likely to be evaluated next.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: reward points with high uncertainty
    # This helps escape local optima
    exploration_weight = 0.3
    exploration_bonus = exploration_weight * sigma_safe
    
    # Cost penalty: penalize expensive points
    # Normalize cost to [0, 1] range approximately
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-8
    cost_normalized = (cost - cost_min) / cost_range
    
    # Penalty scales with how much more expensive the point is
    cost_penalty = 0.2 * cost_normalized
    
    # Combine: EI + exploration - cost penalty
    scores = ei + exploration_bonus - cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
