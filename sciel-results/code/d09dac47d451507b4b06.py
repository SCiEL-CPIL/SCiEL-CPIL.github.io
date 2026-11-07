"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, leave 10s margin
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes)
    weights = np.asarray(weights)
    
    # Reshape if 1D, though contract says (5, 100)
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    # prizes: (5, 100), weights: (5, 100)
    # 1. Normalize weights per constraint to [0, 1]
    # For each constraint, normalize weights so max is 1 and min is 0
    w_min = np.min(weights, axis=1, keepdims=True)  # shape (5, 1)
    w_max = np.max(weights, axis=1, keepdims=True)  # shape (5, 1)
    w_range = w_max - w_min
    w_range = np.where(w_range > 0, w_range, 1.0)  # avoid division by zero
    normalized_weights = (weights - w_min) / w_range  # shape (5, 100), values in [0, 1]
    
    # 2. Calculate the bottleneck normalized weight per item
    # This represents the worst-case relative resource consumption
    max_norm_weight = np.max(normalized_weights, axis=0)  # shape (100,)
    
    # 3. Calculate total prize per item
    prize_sum = np.sum(prizes, axis=0)  # shape (100,)
    
    # 4. Compute raw efficiency score: prize / (1 + bottleneck cost)
    # Using 1 + cost ensures positive denominator and smooth behavior
    raw_scores = prize_sum / (1.0 + max_norm_weight)
    
    # Handle non-finite values in raw scores
    raw_scores = np.where(np.isfinite(raw_scores), raw_scores, 0.0)
    
    # 5. Apply a power transformation to adjust the spread of the heuristic.
    # A power < 1 flattens the distribution, making the heuristic less aggressive.
    # This can help pheromone to guide the search more effectively.
    power = 0.7
    heuristic_scores = raw_scores ** power
    
    # 6. Safety checks
    # Ensure all values are positive and finite
    heuristic_scores = np.where(np.isfinite(heuristic_scores) & (heuristic_scores > 0), heuristic_scores, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        pass # Return whatever we have
        
    return heuristic_scores
# EVOLVE-BLOCK-END
