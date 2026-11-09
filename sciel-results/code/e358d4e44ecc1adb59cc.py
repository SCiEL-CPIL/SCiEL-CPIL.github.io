"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB with a stronger cost penalty.
    
    Parameters:
    - mu: array of shape (2000,) - mean predictions
    - sigma: array of shape (2000,) - standard deviation predictions
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation costs (1 + mean(x))
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term for exploration
    beta = 1.5
    ucb = mu + beta * sigma_safe
    
    # Combine EI and UCB, penalize by cost
    # We want to maximize improvement per unit cost
    # Use a weighted combination: EI dominates, but UCB helps exploration
    # The cost penalty is stronger (beta_cost = 1.5) to prefer low-cost points
    beta_cost = 1.5
    
    # Normalize costs to [0, 1] range for better scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Penalize higher costs more aggressively
    cost_penalty = cost_norm ** beta_cost
    
    # Combine: high EI, high UCB, low cost
    # Scale EI and UCB to comparable ranges
    ei_max = np.max(ei)
    ucb_max = np.max(np.abs(ucb))
    
    if ei_max > 0:
        ei_scaled = ei / ei_max
    else:
        ei_scaled = ei
    
    if ucb_max > 0:
        ucb_scaled = -ucb / ucb_max  # Negate because lower is better for minimization
    else:
        ucb_scaled = -ucb
    
    # Weighted combination: 70% EI, 30% UCB, penalize cost
    scores = 0.7 * ei_scaled + 0.3 * ucb_scaled - 0.5 * cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
