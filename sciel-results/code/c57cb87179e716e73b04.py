"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Uses a multiplicative efficiency metric: product of prizes / product of weights,
    which penalizes items that are heavy in any single constraint more heavily.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Compute product of prizes across constraints (total prize value)
    prize_prod = np.prod(prizes, axis=0)
    
    # Compute product of weights across constraints (combined resource cost)
    weight_prod = np.prod(weights, axis=0)
    
    # Heuristic: prize product / (weight product + epsilon)
    # This gives high scores to items that are light in ALL constraints
    heuristic = prize_prod / (weight_prod + 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
