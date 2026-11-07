"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)
    
    # For each item, compute the 'tightness' - how close it gets to filling
    # each constraint. An item that uses a large fraction of a constraint's
    # capacity is less desirable because it blocks other items.
    # Capacity is 25 for each constraint.
    capacity = 25.0
    
    # Fraction of capacity used in each constraint for each item
    # weights has shape (5, 100), so weights[c, i] is weight of item i in constraint c
    # fraction[c, i] = weights[c, i] / capacity
    fractions = weights / capacity  # shape (5, 100)
    
    # The bottleneck constraint for each item is the one with the highest fraction
    # Items with high max fraction are risky
    max_fraction = fractions.max(axis=0)  # shape (100,)
    
    # Sum of fractions represents total resource usage across all constraints
    sum_fraction = fractions.sum(axis=0)  # shape (100,)
    
    # Heuristic: reward high total prize, penalize both the worst-case constraint
    # usage and overall resource consumption
    # Use a multiplicative penalty that strongly discourages items that are
    # heavy in any single constraint
    # The penalty is (1 + max_fraction)^alpha * (1 + sum_fraction)^beta
    # with alpha and beta tuned to balance the penalties
    
    alpha = 2.0  # strong penalty for bottleneck constraint
    beta = 1.0   # moderate penalty for total resource usage
    
    penalty = (1.0 + max_fraction) ** alpha * (1.0 + sum_fraction) ** beta
    
    # Final heuristic: total_prize / penalty
    # This rewards items with high prize that don't strain any constraint too much
    h = total_prize / penalty
    
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
