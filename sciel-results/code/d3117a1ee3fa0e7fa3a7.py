import numpy as np
import time

# We might need to track time to avoid timeouts, but O(N^3) for N=100 is very fast.
# We will implement the lookahead heuristic.

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Convert to numpy arrays for efficiency
    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)
    
    n = len(weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Calculate densities
    # Avoid division by zero, though weights should be positive
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0)
    
    # Get indices sorted by density descending
    # We only need to consider items that fit in the current capacity as the "first" pick
    # But for the simulation, we need the rest.
    
    best_index = -1
    best_sim_value = -1

    # To speed up, we can limit the number of candidates we simulate for.
    # However, with N=100, simulating all is fine.
    # Let's iterate through all items.
    
    # Pre-compute sorted order of all items by density for the "fill" part
    # This order is used to fill the knapsack after picking a candidate.
    sorted_indices = np.argsort(densities)[::-1]
    
    for i in range(n):
        # Check if item i fits
        if weights[i] > remaining_capacity:
            continue
            
        # Simulate picking item i
        current_value = values[i]
        current_capacity = remaining_capacity - weights[i]
        
        # Fill the rest with greedy density
        for idx in sorted_indices:
            if idx == i:
                continue
            if weights[idx] <= current_capacity:
                current_value += values[idx]
                current_capacity -= weights[idx]
                
        if current_value > best_sim_value:
            best_sim_value = current_value
            best_index = i

    return int(best_index)
