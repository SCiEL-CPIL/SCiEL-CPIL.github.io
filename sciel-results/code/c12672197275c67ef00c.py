"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Computes heuristic scores for items in a Multiple-Constraint Knapsack Problem.
    
    Args:
        prizes: numpy array of shape (5, 100)
        weights: numpy array of shape (5, 100)
        
    Returns:
        numpy array of shape (5, 100) containing heuristic scores.
        The shape is maintained to be compatible with the evaluator's 
        reduction strategy (likely summing or averaging over the first axis).
    """
    # 1. Aggregate prizes. 
    # Assuming the evaluator sums the rows of the heuristic matrix, 
    # we want the effective per-item score to be based on the total prize.
    # If we put the same value in all 5 rows, summing them gives 5 * value.
    # This constant factor doesn't change probabilities, but affects scaling.
    
    # Calculate total prize per item
    total_prize = np.sum(prizes, axis=0)  # Shape: (100,)
    
    # 2. Aggregate weights.
    # Using max weight identifies the most constraining resource for each item.
    # This is often a better predictor of "fit" than sum weight in MKP 
    # because an item that barely fits in one constraint but is light in others 
    # is less risky than an item that is moderately heavy in all.
    max_weight = np.max(weights, axis=0)  # Shape: (100,)
    
    # 3. Calculate raw heuristic: Value Density
    # Higher prize per unit of bottleneck weight is better.
    raw_heuristic = total_prize / (max_weight + 1e-9)  # Shape: (100,)
    
    # 4. Normalize/Scale the heuristic.
    # The pheromone starts at 1.0. If the heuristic is much smaller (e.g., ~0.2),
    # the initial selection is dominated by pheromone (random).
    # If we scale the heuristic so its mean is ~1.0, the heuristic provides
    # significant guidance from the first iteration.
    mean_h = np.mean(raw_heuristic)
    if mean_h > 0:
        scaled_heuristic = raw_heuristic / mean_h
    else:
        scaled_heuristic = raw_heuristic
        
    # Ensure no negative values (prizes/weights are positive, so this is safe, 
    # but good practice)
    scaled_heuristic = np.maximum(scaled_heuristic, 1e-9)
    
    # 5. Broadcast to shape (5, 100) to match input shape and evaluator expectations.
    # The evaluator likely reduces this matrix (e.g., sum or mean over axis 0) 
    # to get the final per-item heuristic.
    heuristic_matrix = np.tile(scaled_heuristic, (5, 1))
    
    return heuristic_matrix
# EVOLVE-BLOCK-END
