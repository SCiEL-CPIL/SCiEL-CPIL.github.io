import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    N_VAL = 256
    # Nodes are 2 to 256 inclusive
    # Map value to index: 2 -> 0, 3 -> 1, ..., 256 -> 254
    values = list(range(2, N_VAL + 1))
    n_nodes = len(values)
    
    # Precompute weights
    weights = [1.0 / v for v in values]
    
    # Precompute conflict masks
    # conflict[i] is a bitmask where bit j is 1 if values[i] and values[j] conflict
    # Conflict if lcm(values[i], values[j]) <= 256
    conflict = [0] * n_nodes
    
    for i in range(n_nodes):
        vi = values[i]
        mask = 0
        for j in range(n_nodes):
            if i == j:
                continue
            vj = values[j]
            # lcm(vi, vj) = vi * vj // gcd(vi, vj)
            g = gcd(vi, vj)
            if vi * vj // g <= N_VAL:
                mask |= (1 << j)
        conflict[i] = mask
        
    # Sort nodes by weight descending to prioritize high value nodes
    # We want to pick the node with the largest weight first in the B&B
    # Let's create an order list: indices sorted by weights descending
    order = sorted(range(n_nodes), key=lambda i: weights[i], reverse=True)
    
    # We will use a recursive B&B.
    # State: 
    #   avail_mask: bitmask of available nodes
    #   curr_sum: current sum of reciprocals
    #   selected_indices: list of indices of selected nodes
    
    global_best_sum = 0.0
    global_best_indices = []
    
    # Heuristic initialization
    # Greedy: pick highest weight, remove conflicts, repeat
    curr_avail = (1 << n_nodes) - 1
    curr_sum = 0.0
    curr_sel = []
    
    for idx in order:
        if (curr_avail >> idx) & 1:
            # Check if it's still available (not removed by previous picks)
            # Since we iterate in order and remove conflicts, if bit is set, it's available
            curr_sum += weights[idx]
            curr_sel.append(idx)
            # Remove conflicts
            curr_avail &= ~conflict[idx]
            curr_avail &= ~(1 << idx)
            
    global_best_sum = curr_sum
    global_best_indices = curr_sel[:]
    
    # B&B
    # To avoid deep recursion stack issues and overhead, we can use a stack or just recursion with N=255.
    # Recursion depth is at most 255, which is fine.
    
    def bound(avail_mask):
        # Upper bound: sum of weights of all available nodes
        # This is a valid bound because we can select at most all of them.
        # We can precompute prefix sums? No, mask changes arbitrarily.
        # Iterating 255 bits is fast.
        s = 0.0
        m = avail_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    def solve(avail_mask, curr_sum, curr_sel):
        nonlocal global_best_sum, global_best_indices
        
        if time.time() - start_time > deadline:
            return
            
        # Pruning
        b = bound(avail_mask)
        if curr_sum + b <= global_best_sum:
            return
            
        if avail_mask == 0:
            if curr_sum > global_best_sum:
                global_best_sum = curr_sum
                global_best_indices = curr_sel[:]
            return
            
        # Pick the first available node in 'order'
        # 'order' is sorted by weight descending.
        # We need to find the first idx in 'order' such that bit is set in avail_mask.
        # To make this fast, we can iterate 'order' and check.
        # Since 'order' is fixed, we can just loop.
        
        chosen_idx = -1
        for idx in order:
            if (avail_mask >> idx) & 1:
                chosen_idx = idx
                break
        
        if chosen_idx == -1:
            return # Should not happen if avail_mask != 0
            
        # Branch 1: Include chosen_idx
        # New available mask: remove chosen_idx and all its conflicts
        new_mask_include = avail_mask & ~conflict[chosen_idx] & ~(1 << chosen_idx)
        new_sum_include = curr_sum + weights[chosen_idx]
        new_sel_include = curr_sel + [chosen_idx]
        
        solve(new_mask_include, new_sum_include, new_sel_include)
        
        # Check time again
        if time.time() - start_time > deadline:
            return
            
        # Branch 2: Exclude chosen_idx
        new_mask_exclude = avail_mask & ~(1 << chosen_idx)
        solve(new_mask_exclude, curr_sum, curr_sel)
        
    # Initial call
    # avail_mask is all nodes
    initial_mask = (1 << n_nodes) - 1
    solve(initial_mask, 0.0, [])
    
    # Construct result
    if not global_best_indices:
        return [256] # Fallback, though greedy should always find something
        
    result_values = [values[i] for i in global_best_indices]
    return result_values

# Test locally? No, just output.
