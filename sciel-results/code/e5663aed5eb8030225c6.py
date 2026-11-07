"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    # Convert to float array to ensure numerical stability
    d = np.array(distance_matrix, dtype=np.float64)
    
    # Add a small epsilon to prevent division by zero
    # The diagonal will be 1e-9, leading to large heuristic values,
    # but these are typically masked out as 'visited' in the ACO loop.
    d_safe = d + 1e-9
    
    # Use a sharper heuristic (alpha=2.0) to emphasize short edges more strongly.
    # This makes the initial ant behavior closer to a Nearest Neighbor greedy approach,
    # which often provides a better starting tour for random uniform points.
    alpha = 2.0
    eta = 1.0 / (d_safe ** alpha)
    
    # Ensure the shape is correct (it is, by construction)
    # Ensure no NaNs or Infs (d_safe is never 0, so no div by zero)
    
    return eta
# EVOLVE-BLOCK-END
