"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Calculate total prize per item
    values = np.sum(prizes, axis=0)
    
    # Calculate total weight per item across all constraints
    total_weights = np.sum(weights, axis=0)
    
    # Heuristic: Value density with smoothed denominator.
    # Using +1.0 instead of +1e-9 smooths the heuristic, preventing
    # extreme values for items with very low weight which might lead
    # the ACO to suboptimal local solutions.
    eta = values / (total_weights + 1.0)
    
    # Ensure positive values as per contract
    eta = np.maximum(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
