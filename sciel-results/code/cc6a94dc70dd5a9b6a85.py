"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes and weights have shape (5, 100)
    # Capacity per constraint is 25
    # We want items that are high prize and low weight across all constraints.
    # A good heuristic is prize / (weight + epsilon) for each constraint, then combine.
    # Since all constraints must be satisfied, we can use the minimum ratio across constraints
    # to ensure no single constraint is a bottleneck, or a weighted sum.
    
    # Let's try: for each item, compute the "efficiency" in each constraint: prize / weight
    # Then take the minimum efficiency across constraints (bottleneck constraint)
    # This ensures we prefer items that are good in ALL constraints, not just one.
    
    # Avoid division by zero
    eps = 1e-9
    ratios = prizes / (weights + eps)  # shape (5, 100)
    
    # Use the minimum ratio across constraints (hardest constraint dominates)
    min_ratios = ratios.min(axis=0)  # shape (100,)
    
    # Also consider total prize / total weight as a secondary factor
    total_prize = prizes.sum(axis=0)  # shape (100,)
    total_weight = weights.sum(axis=0)  # shape (100,)
    
    # Combine: primary is min ratio (bottleneck), secondary is total efficiency
    # Normalize to avoid scale issues
    # Actually, just use min_ratios as the heuristic - it directly reflects the constraint tightness
    
    # But let's blend: items with high min-ratio are preferred
    # Add a small component from total prize/weight for tie-breaking
    total_eff = total_prize / (total_weight + eps)
    
    # Blend the two
    heuristic = 0.7 * min_ratios + 0.3 * total_eff
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
