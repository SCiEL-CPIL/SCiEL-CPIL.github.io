"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute heuristic attractiveness scores for items in a multi-constraint knapsack problem.
    
    Uses a refined multi-objective scoring approach:
    1. Prize efficiency: total prize / sum of weights (cost efficiency)
    2. Bottleneck penalty: penalize items with high weight in any single constraint
    3. Constraint balance: reward items that are uniformly light across constraints
    
    Args:
        prizes: array of shape (5, 100) with uniform values in [0, 1]
        weights: array of shape (5, 100) with uniform values in [0, 1]
    
    Returns:
        Array of shape (100,) with one attractiveness score per item
    """
    # Ensure proper shapes
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Expected shapes (5, 100) for both prizes and weights")
    
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)  # shape: (100,)
    
    # Sum of weights per item (total cost across all constraints)
    sum_w = weights.sum(axis=0)  # shape: (100,)
    
    # Max weight per item (bottleneck constraint)
    max_w = weights.max(axis=0)  # shape: (100,)
    
    # Mean weight per item
    mean_w = weights.mean(axis=0)  # shape: (100,)
    
    # Variance of weights: low variance means the item is balanced across constraints
    w_var = weights.var(axis=0)  # shape: (100,)
    
    # Component 1: Prize per unit total weight (primary efficiency signal)
    # This captures how much prize you get per unit of total resource consumption
    eff_score = total_prize / (sum_w + 1e-9)
    
    # Component 2: Bottleneck penalty
    # Items with very high max_w relative to the capacity (25) are risky
    # Since capacity is 25 and weights are in [0,1], max_w is at most 1
    # But relative to other items, high max_w means it constrains packing
    # Use exponential penalty for items that are heavy in any single constraint
    bottleneck_penalty = np.exp(-2.0 * max_w)
    
    # Component 3: Balance bonus
    # Items with low variance across constraints are more predictable
    # For uniform[0,1], variance ranges from 0 to ~0.0833
    balance_bonus = 1.0 / (1.0 + 8.0 * w_var)
    
    # Combine: primary efficiency scaled by bottleneck resistance and balance
    heuristic = eff_score * bottleneck_penalty * balance_bonus
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
