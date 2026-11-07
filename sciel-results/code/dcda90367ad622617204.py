"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ACO on CVRP.
    
    Uses a power-transformed inverse distance to emphasize short edges,
    which helps ants form tighter clusters and reduce depot returns.
    """
    d = distance_matrix.astype(np.float64)
    # Avoid division by zero; diagonal is 0
    d_safe = np.where(d > 1e-9, d, 1e9)
    # Power > 1 makes short edges disproportionately attractive,
    # encouraging dense local tours that respect capacity constraints
    alpha = 2.0
    heuristic = 1.0 / (d_safe ** alpha)
    # Ensure all entries are positive (diagonal will be tiny but positive)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    return heuristic
# EVOLVE-BLOCK-END
