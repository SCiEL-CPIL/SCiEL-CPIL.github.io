import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    W = np.array(remaining_weights)
    V = np.array(remaining_values)
    C = remaining_capacity
    
    feasible_mask = W <= C
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
        
    densities = V / W
    # Sort indices by density descending
    sorted_indices = np.argsort(densities)[::-1]
    
    best_score = -1
    best_idx = -1
    
    for i in feasible_indices:
        w_i = W[i]
        v_i = V[i]
        c_rem = C - w_i
        
        if c_rem < 0:
            continue
            
        # Calculate LP relaxation value for the remaining items
        # with capacity c_rem, excluding item i
        lp_val = 0.0
        w_acc = 0.0
        
        for j in sorted_indices:
            if j == i:
                continue
            w_j = W[j]
            v_j = V[j]
            
            if w_acc + w_j <= c_rem:
                w_acc += w_j
                lp_val += v_j
            else:
                # Fractional part
                if w_j > 0:
                    frac = (c_rem - w_acc) / w_j
                    lp_val += frac * v_j
                break
                
        total_est = v_i + lp_val
        
        if total_est > best_score:
            best_score = total_est
            best_idx = i
            
    return int(best_idx)
