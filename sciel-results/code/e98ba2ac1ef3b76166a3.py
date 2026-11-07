"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Total prize per item (sum across all 5 constraints)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Total weight per item across all constraints
    total_weight = weights.sum(axis=0)  # shape (100,)
    
    # Max weight per item (bottleneck constraint)
    max_weight = weights.max(axis=0)  # shape (100,)
    
    # Base density: total prize per unit total weight
    density = total_prize / (total_weight + 1e-9)
    
    # Bottleneck penalty: penalize items that are very heavy in any single constraint
    # Normalize by capacity (25) to get relative cost
    bottleneck_ratio = max_weight / 25.0
    bottleneck_penalty = 1.0 / (1.0 + bottleneck_ratio)
    
    # Combine: density weighted by bottleneck penalty
    heuristic = density * bottleneck_penalty
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
