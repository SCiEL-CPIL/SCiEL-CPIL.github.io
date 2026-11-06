import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 1e-12:
        return -1
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    feasible_mask = weights <= remaining_capacity + 1e-12
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Compute densities
    densities = np.zeros(n)
    for i in range(n):
        if weights[i] > 0:
            densities[i] = values[i] / weights[i]
        else:
            densities[i] = float('inf') if values[i] > 0 else 0.0
            
    sorted_indices = np.argsort(-densities)
    
    # Helper to compute fractional knapsack value for a set of items (excluding a specific index)
    # To speed this up, we can pass a list of indices to consider
    def calc_frac_value(cap, exclude_idx, consider_indices):
        val = 0.0
        cur_cap = cap
        for idx in consider_indices:
            if idx == exclude_idx:
                continue
            if cur_cap <= 1e-12:
                break
            w_idx = weights[idx]
            v_idx = values[idx]
            if w_idx <= cur_cap + 1e-12:
                val += v_idx
                cur_cap -= w_idx
            else:
                if w_idx > 0:
                    val += (cur_cap / w_idx) * v_idx
                cur_cap = 0.0
                break
        return val

    # Step 1: 1-lookahead scores
    scores = np.zeros(n)
    for i in feasible_indices:
        w_i = weights[i]
        v_i = values[i]
        new_cap = remaining_capacity - w_i
        if new_cap < -1e-12:
            continue
        # Use all feasible indices except i for the fractional part
        # We pass the sorted feasible indices
        # Optimization: Just iterate sorted_indices and skip i
        frac_val = 0.0
        current_cap = new_cap
        for idx in sorted_indices:
            if idx == i:
                continue
            if current_cap <= 1e-12:
                break
            w_idx = weights[idx]
            v_idx = values[idx]
            if w_idx <= current_cap + 1e-12:
                frac_val += v_idx
                current_cap -= w_idx
            else:
                if w_idx > 0:
                    frac_val += (current_cap / w_idx) * v_idx
                current_cap = 0.0
                break
        scores[i] = v_i + frac_val

    # Get top K candidates
    K = 5
    # Sort feasible indices by score descending
    sorted_feasible = sorted(feasible_indices, key=lambda x: scores[x], reverse=True)
    top_candidates = sorted_feasible[:K]
    
    best_total_score = -1.0
    best_idx = -1
    
    # Step 2: 2-lookahead for top candidates
    for i in top_candidates:
        w_i = weights[i]
        v_i = values[i]
        new_cap = remaining_capacity - w_i
        
        if new_cap < -1e-12:
            continue
            
        best_second_score = -1.0
        
        # Try all other feasible items as the second pick
        for j in feasible_indices:
            if j == i:
                continue
            w_j = weights[j]
            v_j = values[j]
            
            if w_j > new_cap + 1e-12:
                continue
                
            cap_after_j = new_cap - w_j
            
            # Compute fractional value for the rest, excluding i and j
            frac_val = 0.0
            current_cap = cap_after_j
            for idx in sorted_indices:
                if idx == i or idx == j:
                    continue
                if current_cap <= 1e-12:
                    break
                w_idx = weights[idx]
                v_idx = values[idx]
                if w_idx <= current_cap + 1e-12:
                    frac_val += v_idx
                    current_cap -= w_idx
                else:
                    if w_idx > 0:
                        frac_val += (current_cap / w_idx) * v_idx
                    current_cap = 0.0
                    break
                    
            total_second_score = v_j + frac_val
            if total_second_score > best_second_score:
                best_second_score = total_second_score
                
        total_score = v_i + best_second_score
        if total_score > best_total_score:
            best_total_score = total_score
            best_idx = i
            
        if time.time() > deadline:
            break

    if best_idx < 0:
        # Fallback to best 1-step score
        best_idx = sorted_feasible[0] if sorted_feasible else -1

    return int(best_idx) if best_idx >= 0 else -1
