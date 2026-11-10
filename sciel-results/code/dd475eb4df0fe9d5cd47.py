import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Identify feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    # Get indices of feasible items
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    
    # Calculate average density of feasible items
    # Using sum(v)/sum(w) is a robust estimate of marginal value per unit weight
    sum_w = np.sum(feasible_weights)
    sum_v = np.sum(feasible_values)
    
    if sum_w == 0:
        # Fallback to pure value if weights are zero (unlikely in standard knapsack)
        return int(feasible_indices[np.argmax(feasible_values)])
        
    avg_density = sum_v / sum_w
    
    # Calculate projected value for each feasible item
    # Projected Value = Immediate Value + (Remaining Capacity * Avg Density of Remaining Items)
    # Note: The "Remaining Items" for the calculation of item i would ideally exclude i,
    # but for a heuristic, using the global average is a standard approximation (1-step lookahead).
    # A more precise version would subtract i's contribution from the average, but that's O(N^2) or requires careful masking.
    # Let's stick to the global average for O(N) efficiency.
    
    remaining_cap_after = remaining_capacity - feasible_weights
    projected_values = feasible_values + avg_density * remaining_cap_after
    
    # Find the index in the feasible subset with max projected value
    best_local_idx = np.argmax(projected_values)
    
    # Map back to the original index
    best_global_idx = feasible_indices[best_local_idx]
    
    return int(best_global_idx)
