"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes and weights are of shape (5, 100)
    # prizes: 5 constraints/dimensions, 100 items
    # weights: 5 constraints/dimensions, 100 items
    
    # Calculate total prize for each item (sum across the 5 dimensions)
    prize_sum = np.sum(prizes, axis=0)
    
    # Calculate the maximum weight for each item across the 5 constraints
    # This identifies the "bottleneck" resource consumption for each item
    max_weight = np.max(weights, axis=0)
    
    # Heuristic: Total prize divided by the maximum weight consumed in any single constraint
    # This prioritizes items that have high value relative to their most restrictive resource usage
    # and favors "balanced" items that don't heavily burden a single constraint
    heuristic = prize_sum / (max_weight + 1e-9)
    
    # Ensure all values are positive and finite
    # The problem statement says: "Heuristic entries that are not positive become 1e-9"
    # Our formula produces positive values for positive prizes and weights.
    # If prize is 0, heuristic is 0. We should handle 0s?
    # "Heuristic entries that are not positive become 1e-9"
    # So if prize_sum is 0, heuristic is 0, which becomes 1e-9.
    
    heuristic[heuristic <= 0] = 1e-9
    heuristic[~np.isfinite(heuristic)] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
