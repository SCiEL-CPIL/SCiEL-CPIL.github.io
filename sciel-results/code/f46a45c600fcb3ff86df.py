import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    n = len(w)
    
    if n == 0:
        return -1
        
    # Calculate densities
    # Handle zero weights (should not happen, but safe)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(w > 0, v / w, 0)
        
    # Sort indices by density descending
    sorted_indices = np.argsort(-densities)
    
    support = []
    current_cap = remaining_capacity
    
    # Determine the set of items included in the optimal fractional solution
    for idx in sorted_indices:
        if w[idx] <= current_cap:
            support.append(idx)
            current_cap -= w[idx]
        else:
            # This item is partially included in the fractional solution
            support.append(idx)
            break
            
    if not support:
        return -1
        
    # Select the item from the support with the lowest weight
    # Tie-breaker: highest value
    # We want min weight, so we can sort support by weight asc, then value desc
    # Or just iterate and keep track
    
    best_idx = support[0]
    min_weight = w[best_idx]
    max_value = v[best_idx]
    
    for idx in support[1:]:
        if w[idx] < min_weight:
            min_weight = w[idx]
            max_value = v[idx]
            best_idx = idx
        elif w[idx] == min_weight:
            if v[idx] > max_value:
                max_value = v[idx]
                best_idx = idx
                
    return int(best_idx)
