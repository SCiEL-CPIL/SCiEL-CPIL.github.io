import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0  # 120s margin out of 150s total
    
    # Convert to numpy arrays for vectorization
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    C = float(remaining_capacity)
    
    # Filter feasible items (weight <= capacity)
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    w_f = w[feasible_mask]
    v_f = v[feasible_mask]
    n_f = len(w_f)
    
    if n_f == 0:
        return -1
        
    # Compute densities
    d_f = np.zeros(n_f)
    non_zero_w = w_f > 0
    d_f[non_zero_w] = v_f[non_zero_w] / w_f[non_zero_w]
    d_f[~non_zero_w] = np.where(v_f[~non_zero_w] > 0, np.inf, 0.0)
    
    # Sort indices by density descending
    sort_idx = np.argsort(-d_f)
    w_s = w_f[sort_idx]
    v_s = v_f[sort_idx]
    d_s = d_f[sort_idx]
    
    # Prefix sums of weights and values in sorted order
    pw = np.cumsum(w_s)
    pv = np.cumsum(v_s)
    
    def get_lp_value(cap):
        """
        Calculate the fractional knapsack value for capacity 'cap'
        using the pre-sorted items w_s, v_s, d_s.
        """
        if cap <= 1e-9:
            return 0.0
        
        # Find the largest index k such that pw[k] <= cap
        # searchsorted returns the index where cap would be inserted to maintain order
        # side='right' means it returns index i such that a[i-1] <= cap < a[i]
        k = np.searchsorted(pw, cap, side='right') - 1
        
        if k < 0:
            # Cap is less than the weight of the first item
            # We take a fraction of the first item
            if len(w_s) == 0:
                return 0.0
            # Value is proportional to weight
            return cap * d_s[0]
        
        val = pv[k]
        remaining_cap = cap - pw[k]
        
        # If there is a next item, we take a fraction of it
        if k < len(w_s) - 1:
            next_w = w_s[k + 1]
            next_d = d_s[k + 1]
            # We can take up to next_w of the next item
            # But we only have remaining_cap capacity
            # Fractional value = remaining_cap * density
            # (Since we are in fractional mode, we assume we can take any fraction)
            val += remaining_cap * next_d
            
        return val
    
    def greedy_simulate(cap, exclude_set):
        """
        Simulate greedy selection by density, skipping items in exclude_set.
        Returns the total value obtained.
        """
        total_val = 0.0
        remaining_cap = cap
        
        # Iterate through items in density order
        for idx in range(len(w_s)):
            if remaining_cap <= 1e-9:
                break
            
            # Check if this item is in the exclude set
            # We need to map sorted index back to original feasible index
            orig_idx = sort_idx[idx]
            if orig_idx in exclude_set:
                continue
                
            item_w = w_s[idx]
            item_v = v_s[idx]
            
            if item_w <= remaining_cap:
                remaining_cap -= item_w
                total_val += item_v
            else:
                # In 0/1 knapsack, we can't take fractions, so we skip
                # But for a greedy simulation heuristic, we might just stop
                # or continue trying smaller items. Standard greedy stops.
                # However, since we are using this as a *score* for the lookahead,
                # and the actual subsequent calls will be greedy, we should simulate
                # the actual greedy behavior.
                # Standard greedy: if item doesn't fit, skip it and try next?
                # No, usually greedy by density checks if it fits. If not, it might skip.
                # Let's assume standard greedy: iterate through sorted list, take if fits.
                pass
        
        return total_val

    best_score = -1.0
    best_i_local = -1
    
    # 2-step lookahead with 1-step greedy simulation of remainder
    # For each candidate first item i, and second item j (j != i, w_i + w_j <= C),
    # score = v_i + v_j + greedy_simulate(C - w_i - w_j, {i, j})
    
    # Optimization: Pre-compute greedy simulation for all possible remainders?
    # No, exclude_set changes.
    
    # Since N=100, N^2 = 10,000 pairs.
    # Greedy simulation is O(N). Total O(N^3) = 1,000,000 ops.
    # This is fast enough in Python if optimized, or NumPy.
    
    # To speed up, we can pre-compute the greedy path for the full set,
    # but exclusions make it complex.
    
    # Let's try a simpler 1-step lookahead first as per inspiration 1 (score 1.0015)
    # "1-step lookahead with full greedy simulation of the remainder"
    # This means: for each i, score = v_i + greedy_simulate(C - w_i, {i})
    
    # Let's implement 1-step lookahead with greedy simulation first, as it's O(N^2)
    # and measured to be better than pure greedy.
    
    for i in range(n_f):
        if w_f[i] > C:
            continue
        
        rem_cap = C - w_f[i]
        # Simulate greedy on remaining items, excluding i
        # We can do this by iterating through sorted items and skipping i
        
        sim_val = 0.0
        current_cap = rem_cap
        for idx in range(len(w_s)):
            if current_cap <= 1e-9:
                break
            orig_idx = sort_idx[idx]
            if orig_idx == i:
                continue
            item_w = w_s[idx]
            item_v = v_s[idx]
            if item_w <= current_cap:
                current_cap -= item_w
                sim_val += item_v
        
        score = v_f[i] + sim_val
        
        if score > best_score:
            best_score = score
            best_i_local = i
            
        # Check time
        if time.time() > deadline:
            break

    # Fallback if no item found (should not happen if n_f > 0 and w_f[i] <= C)
    if best_i_local == -1:
        for i in range(n_f):
            if w_f[i] <= C:
                best_i_local = i
                break

    if best_i_local == -1:
        return -1

    return int(feasible_indices[best_i_local])
