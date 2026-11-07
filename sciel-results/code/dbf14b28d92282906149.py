"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    # Use inverse distance with a moderate power to balance greediness
    # and exploration. Power 1.5 is less extreme than 2, allowing
    # pheromone to have more influence in later iterations while still
    # preferring shorter edges.
    eps = 1e-9
    d = distance_matrix + eps
    eta = 1.0 / (d ** 1.5)
    
    # Handle any potential non-finite values
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
