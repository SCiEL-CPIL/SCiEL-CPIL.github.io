"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes and weights are expected to be shape (5, 100)
    # Ensure they are numpy arrays
    prizes = np.asarray(prizes)
    weights = np.asarray(weights)
    
    # Calculate total prize for each item (sum over the 5 components)
    total_prizes = prizes.sum(axis=0)
    
    # Calculate the maximum weight component for each item.
    # This penalizes items that are heavy in any single constraint,
    # which is crucial for multi-constraint knapsack problems to avoid
    # exhausting one constraint while leaving others underutilized.
    max_weights = weights.max(axis=0)
    
    # Heuristic: Value to Max-Weight ratio.
    # This favors items that provide high value relative to their
    # most restrictive weight dimension.
    epsilon = 1e-9
    scores = total_prizes / (max_weights + epsilon)
    
    # Ensure the output is a 1D array of floats
    return scores
# EVOLVE-BLOCK-END
