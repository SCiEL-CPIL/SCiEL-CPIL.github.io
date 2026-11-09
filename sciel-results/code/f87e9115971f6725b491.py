import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Filter feasible items? No, we need to return index into original array.
    # We only consider feasible items as candidates for the 'next' pick.
    
    # 1. Compute densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    densities_safe = np.nan_to_num(densities, nan=0.0, posinf=1e18, neginf=0.0)
    
    # Sort indices by density descending for the remainder estimation
    sorted_indices = np.argsort(-densities_safe)
    
    best_score = -1
    best_idx = -1
    
    for i in range(n):
        wi = w[i]
        if wi > c:
            continue
            
        vi = v[i]
        c_new = c - wi
        
        # Estimate value of remaining items using Fractional Knapsack
        # This is an upper bound, often a better guide than 0/1 greedy
        frac_val = 0.0
        cap_left = c_new
        
        for idx in sorted_indices:
            if idx == i:
                continue
                
            wj = w[idx]
            vj = v[idx]
            
            if wj <= cap_left:
                frac_val += vj
                cap_left -= wj
            else:
                # Take a fraction of this item
                if wj > 0:
                    frac_val += vj * (cap_left / wj)
                cap_left = 0
                break
                
        total_score = vi + frac_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            
    return best_idx
