import numpy as np
import time

def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    # Capacity per constraint is 25
    capacity = 25.0
    
    # For each item, compute the fraction of capacity consumed in each constraint
    # weight_fraction[c, i] = weights[c, i] / capacity
    weight_fractions = weights / capacity  # shape (num_constraints, num_items)
    
    # The bottleneck for each item is the constraint where it consumes the most capacity
    max_fraction = np.max(weight_fractions, axis=0)  # shape (num_items,)
    
    # Total prize for each item (sum across constraints)
    total_prize = np.sum(prizes, axis=0)  # shape (num_items,)
    
    # Heuristic: total prize divided by the worst-case capacity fraction
    # This rewards items with high total prize and low worst-case resource usage
    eps = 1e-9
    heuristic = total_prize / (max_fraction + eps)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
