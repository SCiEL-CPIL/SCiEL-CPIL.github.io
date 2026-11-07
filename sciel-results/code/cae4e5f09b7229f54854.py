"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Strategy: Combine inverse distance with a degree-based penalty to encourage
    tour-like structures. We use a soft approach:
    1. Base heuristic: 1 / (distance + eps)
    2. Adjust for demand: edges to high-demand nodes are slightly less attractive
       if they are far, to balance load.
    3. Use a power to sharpen the difference between good and bad edges.
    """
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Ensure we work with floats
    dist = distance_matrix.astype(float)
    
    # Base heuristic: inverse distance
    # We add a small offset to avoid division by zero and to allow for
    # a more nuanced scaling
    base = 1.0 / (dist + 0.1)
    
    # Demand-aware adjustment:
    # Nodes with higher demand are 'harder' to serve if far away.
    # We can penalize edges to high-demand nodes slightly more if distance is large.
    # However, since demand is uniform 1-9, the effect might be small.
    # Let's try a simpler approach: sharpen the heuristic to focus on local moves.
    
    # Apply a power to sharpen. Power > 1 makes short edges much more attractive.
    # This helps the ACO find tight routes.
    power = 2.0
    heuristic = base ** power
    
    # Ensure no zeros or negatives
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
# EVOLVE-BLOCK-END
