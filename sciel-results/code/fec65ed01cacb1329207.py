"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes heuristic scores for items in a Multiple-Constraint Knapsack Problem.
    Uses a numerically stable soft-min of the prize-to-weight ratios across constraints.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure inputs are valid shapes
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        return np.ones(100)
        
    # Avoid division by zero
    eps = 1e-9
    safe_weights = weights + eps
    
    # Calculate ratios for each constraint and item
    # Shape: (5, 100)
    ratios = prizes / safe_weights
    
    # Calculate soft-min across the 5 constraints for each item
    # softmin(x) = min(x) - log(sum(exp(-(x - min(x)))))
    # This is equivalent to -log(sum(exp(-x))) but numerically stable.
    
    # min_ratio has shape (100,)
    min_ratio = np.min(ratios, axis=0)
    
    # Subtract min to stabilize exponentials
    # Shape: (5, 100)
    shifted_ratios = ratios - min_ratio[None, :]
    
    # exp of negative shifted ratios
    # Shape: (5, 100)
    exp_terms = np.exp(-shifted_ratios)
    
    # Sum across constraints
    # Shape: (100,)
    sum_exp = np.sum(exp_terms, axis=0)
    
    # Log of sum
    log_sum_exp = np.log(sum_exp)
    
    # Soft-min
    heuristic = min_ratio - log_sum_exp
    
    # Ensure heuristic is positive and finite
    # If any values are not finite, replace with 1e-9
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
