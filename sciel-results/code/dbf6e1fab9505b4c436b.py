"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes and weights are expected to be shape (5, 100)
    # Ensure they are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Global Ratio Heuristic: Total Prize / Total Weight
    # This favors items that give high total prize for low total resource consumption
    # across all constraints.
    
    # Sum prizes and weights across all constraints for each item
    total_prize = prizes.sum(axis=0)
    total_weight = weights.sum(axis=0)
    
    # Calculate the ratio, adding epsilon to avoid division by zero
    epsilon = 1e-9
    scores = total_prize / (total_weight + epsilon)
    
    # Ensure the output is a 1D array of floats
    return scores
# EVOLVE-BLOCK-END
