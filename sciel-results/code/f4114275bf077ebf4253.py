import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack using a Lagrangian heuristic.
    The score for each item is v_i - lambda * w_i, where lambda is the marginal
    density from the fractional knapsack upper bound.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Convert to numpy arrays for speed
    w = np.array(remaining_weights, dtype=float)
    v = np.array(remaining_values, dtype=float)
    
    # Filter for feasible items (weight <= capacity)
    # Note: The prompt says "remaining (feasible) arrays", but it's safer to check.
    feasible_mask = w <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    w_feas = w[feasible_mask]
    v_feas = v[feasible_mask]
    indices_feas = np.where(feasible_mask)[0]
    
    n_feas = len(w_feas)
    
    # Calculate densities
    densities = v_feas / w_feas
    
    # Sort by density descending to find the fractional knapsack marginal density
    sorted_indices = np.argsort(-densities)
    sorted_w = w_feas[sorted_indices]
    sorted_v = v_feas[sorted_indices]
    sorted_dens = densities[sorted_indices]
    
    # Find the marginal density lambda
    # We simulate filling the knapsack
    current_weight = 0.0
    lambda_val = 0.0
    
    for i in range(n_feas):
        if current_weight + sorted_w[i] <= remaining_capacity:
            current_weight += sorted_w[i]
            # If this was the last item that fit, the next one determines lambda
            # But we need the density of the item that *would* be next or the fractional one.
            # Actually, if we fit this item, and there are more items, lambda is the density of the NEXT item.
            # If this is the last item (i == n_feas - 1), lambda is 0 (no more items).
            if i == n_feas - 1:
                lambda_val = 0.0
                break
        else:
            # This item does not fit fully. It is the fractional item.
            # Its density is the marginal value of capacity.
            lambda_val = sorted_dens[i]
            break
    else:
        # Loop finished without break, means all items fit.
        lambda_val = 0.0

    # Calculate scores: v_i - lambda * w_i
    scores = v_feas - lambda_val * w_feas
    
    # Find the best item
    best_idx_in_feas = np.argmax(scores)
    
    # Check if the best score is positive (worth taking)
    if scores[best_idx_in_feas] <= 0:
        return -1
        
    # Map back to original index
    return int(indices_feas[best_idx_in_feas])
