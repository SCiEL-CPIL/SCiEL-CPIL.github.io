import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    n = len(w)
    
    if n == 0:
        return -1
        
    # Identify feasible items
    feasible_mask = w <= remaining_capacity
    
    if not np.any(feasible_mask):
        return -1
        
    # Get indices of feasible items
    feasible_indices = np.where(feasible_mask)[0]
    
    # Get weights and values for feasible items
    feasible_w = w[feasible_indices]
    feasible_v = v[feasible_indices]
    
    # Filter out zero weights
    valid_mask = feasible_w > 0
    if not np.any(valid_mask):
        return -1
        
    feasible_indices = feasible_indices[valid_mask]
    feasible_w = feasible_w[valid_mask]
    feasible_v = feasible_v[valid_mask]
    
    # Calculate densities
    densities = feasible_v / feasible_w
    
    # Find the maximum density
    max_density = np.max(densities)
    
    # Find all items with this maximum density
    # Using a small epsilon to handle floating point precision if needed, 
    # but exact match is usually fine for generated integer-like data converted to float.
    # To be safe against float precision, we can check close to max.
    # However, if values are random floats, exact ties are unlikely.
    # If values are integers, ties are possible.
    
    # Let's stick to strict max for simplicity, but handle ties by value.
    # We want the index in feasible_indices corresponding to the best choice.
    
    # If there are multiple items with max density, pick the one with the highest value.
    # This is a heuristic improvement over arbitrary selection.
    
    # Create a mask for items with max density
    # Using np.isclose to be robust
    max_density_mask = np.isclose(densities, max_density)
    
    if not np.any(max_density_mask):
        # Fallback, shouldn't happen
        best_local_idx = np.argmax(densities)
    else:
        # Get indices of items with max density
        candidates = np.where(max_density_mask)[0]
        
        if len(candidates) == 1:
            best_local_idx = candidates[0]
        else:
            # Among candidates, pick the one with the highest value
            candidate_values = feasible_v[candidates]
            best_candidate_local_idx = candidates[np.argmax(candidate_values)]
            best_local_idx = best_candidate_local_idx
            
    best_global_idx = feasible_indices[best_local_idx]
    
    return int(best_global_idx)
