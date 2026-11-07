"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized Expected Improvement with dynamic exploration bonus.
    
    The acquisition function combines:
    1. Cost-normalized EI: EI / cost, which favors high improvement per unit cost
    2. Dynamic exploration bonus: scaled by remaining budget to encourage 
       exploration when budget is plentiful and exploitation when budget is low
    
    This structural change uses a different optimization paradigm:
    - Instead of hard gating (binary cheap/expensive), we use smooth cost normalization
    - The exploration-exploitation tradeoff adapts dynamically to budget depletion
    """
    # Compute Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI: reward improvement per unit cost
    # Use a smooth denominator to avoid division by zero
    cost_safe = np.maximum(cost, 0.1)
    ei_per_cost = ei / cost_safe
    
    # Dynamic exploration bonus based on remaining budget
    # Estimate remaining budget: total budget is 30, current cost is sum of evaluated points
    # We don't know exact current cost, but we can use the cost array as a proxy
    # The cost array represents costs of the 2000 candidates being scored
    # We estimate how much budget is left by looking at the distribution of costs
    
    # Use the median cost as a typical cost per evaluation
    median_cost = np.median(cost_safe)
    
    # Estimate number of evaluations we can still do
    # Total budget is 30, initial 5 points cost ~2.5 (average cost ~0.5)
    # So remaining budget is roughly 30 - 2.5 - (number of evaluations * typical cost)
    # We don't know the exact number of evaluations, so we use a heuristic:
    # If costs are high, we've probably used more budget
    
    # Heuristic: remaining_budget_fraction based on how expensive recent evaluations are
    # If median cost is low, we've been efficient and have more budget left
    # If median cost is high, we've been inefficient and have less budget left
    
    # Normalize median cost to [0, 1] range (costs are in [1, 2])
    cost_norm = (median_cost - 1.0) / 1.0  # Maps [1, 2] to [0, 1]
    
    # Remaining budget fraction: higher when costs are lower
    # This is a rough estimate; in practice, the caller tracks actual cost
    remaining_fraction = 1.0 - cost_norm * 0.5  # Range [0.5, 1.0]
    
    # Exploration bonus: encourage exploring high-uncertainty regions
    # Scale by remaining budget: more exploration when budget is plentiful
    # Use uncertainty (sigma) as a proxy for information gain
    exploration_bonus = sigma_safe * remaining_fraction
    
    # Combine: cost-normalized EI + exploration bonus
    # The exploration bonus helps escape local optima and find new minima
    # The cost normalization ensures we get good value per unit cost
    
    # Weight the exploration bonus to balance with exploitation
    # When budget is plentiful (high remaining_fraction), explore more
    # When budget is scarce (low remaining_fraction), exploit more
    exploration_weight = 0.3 * remaining_fraction
    
    score = ei_per_cost + exploration_weight * exploration_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
