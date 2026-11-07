"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(distance_matrix):
    """
    Computes an edge-attractiveness matrix for ACO.
    Uses inverse squared distance to emphasize short edges more strongly.
    """
    # Add a small epsilon to avoid division by zero
    eps = 1e-9
    # Use power 2 to make the heuristic more sensitive to short distances
    # This penalizes long edges more heavily, guiding ants towards local clusters
    return 1.0 / (distance_matrix + eps) ** 2
# EVOLVE-BLOCK-END
