import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # Convert to numpy arrays for easier handling
    W = np.array(remaining_weights)
    V = np.array(remaining_values)
    C = remaining_capacity
    
    # Check for feasible items
    feasible_mask = W <= C
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
        
    # Calculate densities
    # Avoid division by zero, though weights should be > 0
    densities = V / W
    
    # Sort all indices by density descending
    # This order is used for the greedy completion simulation
    sorted_indices = np.argsort(densities)[::-1]
    
    best_score = -1
    best_idx = -1
    
    # For each feasible candidate item i
    for i in feasible_indices:
        w_i = W[i]
        v_i = V[i]
        c_rem = C - w_i
        
        current_score = v_i
        
        # Simulate the standard greedy algorithm for the remaining items
        # We iterate through items in density order.
        # If an item fits, we take it. If not, we skip it (do not break).
        # This estimates the actual value achievable by the greedy path starting with i.
        for j in sorted_indices:
            if j == i:
                continue
            w_j = W[j]
            if w_j <= c_rem:
                current_score += V[j]
                c_rem -= w_j
            # else: skip this item, continue checking others
            # Note: We do not break here because a later item might have a smaller weight and fit.
            
        if current_score > best_score:
            best_score = current_score
            best_idx = i
            
    return int(best_idx)
