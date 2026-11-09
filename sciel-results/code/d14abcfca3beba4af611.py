"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 2.0


def _check_time():
    return (time.time() - _START_TIME) < (_DEADLINE - _MARGIN)


def _vectorized_greedy_fill(capacity, weights, values, exclude_indices=None):
    """
    Simulates greedy fill by density for the given arrays, excluding specified indices.
    Returns the total value achieved.
    """
    if capacity <= 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
        
    # Create mask of items not excluded
    if exclude_indices is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_indices] = False
    else:
        mask = np.ones(n, dtype=bool)
        
    feasible_mask = (weights <= capacity) & mask
    if not np.any(feasible_mask):
        return 0.0
        
    fw = weights[feasible_mask]
    fv = values[feasible_mask]
    
    if len(fw) == 0:
        return 0.0
        
    densities = fv / fw
    order = np.argsort(-densities)
    
    total_val = 0.0
    remaining_cap = capacity
    
    for idx in order:
        w = fw[idx]
        if w <= remaining_cap:
            remaining_cap -= w
            total_val += fv[idx]
        else:
            # Since sorted by density, if this one doesn't fit, 
            # subsequent ones (lower density) might still fit if they are smaller? 
            # No, greedy by density doesn't guarantee smaller weight next.
            # We must check all.
            pass
            
    # The loop above is inefficient if we just break. 
    # Actually, for a proper greedy simulation, we iterate all.
    # But we can vectorize the "can fit" check? No, it's sequential due to capacity reduction.
    
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    if remaining_capacity <= 0:
        return -1

    # Filter to feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_indices]
    fv = remaining_values[feasible_indices]
    nf = len(fw)

    if nf == 0:
        return -1

    # Compute densities for sorting
    densities = fv / fw
    order = np.argsort(-densities)
    
    # Strategy: 2-Step Lookahead
    # For each candidate i in the first few best density items (to limit cost),
    # simulate:
    # 1. Pick i.
    # 2. From remaining, pick the best j (by density) that fits.
    # 3. Fill rest greedily.
    # We restrict candidates to the top K items by density to keep O(K * N^2) manageable,
    # but we verify the global best among them.
    
    # To be safe and fast, we evaluate the top 20 candidates by density.
    # If the optimal item is not in the top 20 density items, it's likely a "value" monster
    # with low density. We should also consider top 5 by value.
    
    candidate_indices = np.unique(np.concatenate([
        order[:20], 
        np.argsort(-fv)[:5]
    ]))
    
    best_score = -1.0
    best_idx = -1
    
    for i in candidate_indices:
        if not _check_time():
            break
            
        # Exclude item i
        exclude_i = i
        
        # Step 1: Value of i
        val_i = fv[i]
        cap_after_i = remaining_capacity - fw[i]
        
        if cap_after_i <= 0:
            # If no capacity left, score is just val_i
            if val_i > best_score:
                best_score = val_i
                best_idx = feasible_indices[i]
            continue
            
        # Step 2: Find best j (by density) from remaining items (excluding i) that fits in cap_after_i
        # We need to simulate the greedy fill after picking i.
        # But to do 2-step, we need to pick the *first* item of the fill.
        # The greedy fill picks the highest density item that fits.
        
        # Let's find the highest density item j != i that fits in cap_after_i.
        # Note: j must be from the feasible set of the *original* list, but 
        # its feasibility depends on cap_after_i.
        
        # To optimize, we can pre-compute the sorted order of all items by density.
        # The greedy fill will pick items in that order, skipping those that don't fit.
        
        # For the 2nd step, we just need to know: if we pick i, what is the value of the 
        # greedy fill of the remaining capacity?
        # This is exactly what the current program does (1-step lookahead = pick i, then greedy fill).
        
        # The difference in 2-step:
        # The current program assumes the *next* call will also do a 1-step lookahead.
        # If we do 2-step lookahead here, we are assuming the *next* call will pick the 
        # item that is best in the context of *this* 2-step simulation.
        # This creates a consistency if the heuristic is "Pick the item that leads to the best 
        # 2-step outcome".
        
        # Let's implement the simulation:
        # 1. Remove i.
        # 2. Find the first item j in density order (excluding i) that fits in cap_after_i.
        # 3. Pick j.
        # 4. Fill rest greedily.
        
        # To find j efficiently:
        # We have the global density order 'order'.
        # We iterate 'order' to find the first idx != i such that fw[idx] <= cap_after_i.
        
        j_found = -1
        cap_after_j = cap_after_i
        val_j = 0.0
        
        for j_idx in order:
            if j_idx == i:
                continue
            if fw[j_idx] <= cap_after_i:
                j_found = j_idx
                val_j = fv[j_idx]
                cap_after_j = cap_after_i - fw[j_idx]
                break
        
        # Now simulate greedy fill for the rest, excluding i and j_found
        if j_found != -1:
            exclude_set = {i, j_found}
            # We need to simulate greedy fill on the remaining capacity cap_after_j
            # excluding i and j_found.
            
            sim_val = val_i + val_j
            rem_cap = cap_after_j
            
            for k_idx in order:
                if k_idx in exclude_set:
                    continue
                if fw[k_idx] <= rem_cap:
                    rem_cap -= fw[k_idx]
                    sim_val += fv[k_idx]
                if rem_cap <= 0:
                    break
        else:
            # No j fits? Then just fill rest with i excluded.
            # This case is rare if i is small, but possible if i takes all capacity.
            # Actually if i takes all capacity, cap_after_i is 0, handled above.
            # If cap_after_i > 0 but no single item fits, we can't pick any more items.
            sim_val = val_i
            # Wait, if no item fits in cap_after_i, greedy fill returns 0.
            
        if sim_val > best_score:
            best_score = sim_val
            best_idx = feasible_indices[i]

    return int(best_idx)
# EVOLVE-BLOCK-END
