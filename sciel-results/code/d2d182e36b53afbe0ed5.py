"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    # Add a small epsilon to avoid division by zero
    d = distance_matrix + 1e-9
    # Use a sharper heuristic (inverse square) to strongly bias ants towards shorter edges
    # This reduces the probability of selecting suboptimal longer edges early in the tour construction
    return 1.0 / (d ** 2)
# EVOLVE-BLOCK-END
