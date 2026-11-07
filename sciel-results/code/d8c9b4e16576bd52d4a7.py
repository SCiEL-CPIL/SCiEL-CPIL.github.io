"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: combine expected improvement with uncertainty
    # and penalize expensive evaluations.
    #
    # mu, sigma: shape (2000,) - GP mean and std in objective units
    # f_best: scalar - best observed value
    # cost: shape (2000,) - cost of evaluating each candidate (1 + mean(x_scaled))
    #
    # Strategy:
    # 1. Compute z-score for expected improvement.
    # 2. Compute EI per unit cost.
    # 3. Add an exploration bonus scaled by uncertainty to encourage
    #    visiting unexplored regions.
    # 4. Handle edge cases (zero sigma, negative improvement).
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware EI
    # cost is in [1, 2] since it's 1 + mean(x_scaled) where x_scaled in [0,1]
    # Higher cost means more expensive, so divide by cost
    cost_safe = np.maximum(cost, 1e-8)
    ei_per_cost = ei / cost_safe
    
    # Exploration bonus: reward high uncertainty regions
    # This helps escape local optima in rastrigin's many local minima
    # Scale the exploration term relative to the typical EI magnitude
    # Use a moderate exploration weight
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    
    # Combine exploitation (EI) and exploration
    scores = ei_per_cost + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores (higher is better)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
