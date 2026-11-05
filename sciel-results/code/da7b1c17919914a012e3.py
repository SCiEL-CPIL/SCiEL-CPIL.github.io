import numpy as np
import time
import random

# Global timer
_start_time = time.time()
_TIMEOUT = 140.0 # Safety margin before 150s limit

def _greedy_solve(w, v, C, order):
    """
    Greedy fill based on a specific order.
    Returns (total_value, total_weight, indices_list)
    """
    current_indices = []
    current_value = 0.0
    current_weight = 0.0
    
    for idx in order:
        if w[idx] <= C - current_weight:
            current_indices.append(idx)
            current_value += v[idx]
            current_weight += w[idx]
            
    return current_value, current_weight, current_indices

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: Simple greedy by density
        if len(remaining_weights) == 0:
            return -1
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v[feasible] / w[feasible]
        # Handle inf/nan by replacing with 0 or large number
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # Convert to numpy arrays for speed
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity
    
    # Ensure no zero weights to avoid division issues, though problem implies positive weights.
    # If zero weight, value is free.
    zero_weight_mask = w == 0
    if np.any(zero_weight_mask):
        # Just take all zero weight items first? 
        # The interface asks for ONE item to add. 
        # If we have zero weight items, they are always feasible.
        # Greedy by value is best for zero weight.
        # But let's stick to the main logic. If w=0, density is inf.
        pass

    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    # Replace inf with a large number for sorting stability, nan with 0
    densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
    
    # 1. Initial Greedy Solutions with different keys
    # Key 1: Density
    order_density = np.argsort(-densities, kind='stable')
    val_d, wt_d, idx_d = _greedy_solve(w, v, C, order_density)
    
    # Key 2: Value
    order_value = np.argsort(-v, kind='stable')
    val_v, wt_v, idx_v = _greedy_solve(w, v, C, order_value)
    
    # Key 3: Weight (ascending) - often good for packing tight
    order_weight = np.argsort(w, kind='stable')
    val_w, wt_w, idx_w = _greedy_solve(w, v, C, order_weight)
    
    # Key 4: Value/Weight^2 (sometimes helps with distribution)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities_sq = v / (w * w)
    densities_sq = np.nan_to_num(densities_sq, nan=0.0, posinf=1e9, neginf=-1e9)
    order_sq = np.argsort(-densities_sq, kind='stable')
    val_sq, wt_sq, idx_sq = _greedy_solve(w, v, C, order_sq)

    # Pick the best initial solution
    candidates = [
        (val_d, idx_d),
        (val_v, idx_v),
        (val_w, idx_w),
        (val_sq, idx_sq)
    ]
    
    best_val, best_idx_list = max(candidates, key=lambda x: x[0])
    
    if not best_idx_list:
        return -1
        
    current_indices = list(best_idx_list)
    current_value = best_val
    current_weight = sum(w[i] for i in current_indices)
    
    sol_set = set(current_indices)
    out_set = set(range(n)) - sol_set
    
    # 2. Local Search
    # We will perform 1-for-1, 1-for-2, and 2-for-1 swaps.
    # To keep it fast, we limit the number of full passes.
    
    max_passes = 20
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        if time.time() - _start_time > _TIMEOUT - 10:
            break
            
        improved = False
        passes += 1
        
        # Pre-compute sorted lists for efficiency in inner loops
        # Sort out_set by density descending (likely good candidates to add)
        out_list = list(out_set)
        if out_list:
            out_list.sort(key=lambda i: -densities[i])
        else:
            out_list = []
            
        # Sort sol_set by density ascending (likely bad candidates to remove)
        sol_list = list(sol_set)
        if sol_list:
            sol_list.sort(key=lambda i: densities[i])
        else:
            sol_list = []

        # --- 1-for-1 Swaps ---
        # Try to replace one item in sol with one item in out
        # We iterate through sol items (worst first) and out items (best first)
        # To avoid O(N^2) full scan every time, we can limit or use heuristics.
        # Given N=100, O(N^2) is 10,000 ops. Very fast.
        
        found_11 = False
        for i_in in sol_list:
            if found_11:
                break
            # If i_in is very good, maybe skip? No, we sorted worst first.
            for i_out in out_list:
                # Check if swap is valid weight-wise
                delta_w = w[i_out] - w[i_in]
                if current_weight + delta_w <= C:
                    delta_v = v[i_out] - v[i_in]
                    if delta_v > 1e-9:
                        # Accept swap
                        sol_set.remove(i_in)
                        sol_set.add(i_out)
                        out_set.remove(i_out)
                        out_set.add(i_in)
                        
                        current_weight += delta_w
                        current_value += delta_v
                        
                        improved = True
                        found_11 = True
                        # Restart loops as sets changed
                        break
            
        if found_11:
            continue

        # --- 1-for-2 Swaps (Remove 1, Add 2) ---
        # This can be beneficial if we have space but low value density items.
        # Or if we want to add two high value items that don't fit individually with current items.
        # Actually, 1-for-2 means: Remove i_in, Add i_out1, i_out2.
        # Condition: w[i_out1] + w[i_out2] - w[i_in] <= C - current_weight + w[i_in]
        #           => w[i_out1] + w[i_out2] <= C - current_weight + w[i_in]
        # Value Gain: v[i_out1] + v[i_out2] - v[i_in] > 0
        
        found_12 = False
        # Iterate sol items (worst first)
        for i_in in sol_list:
            if found_12:
                break
            # Space available after removing i_in
            space_avail = C - (current_weight - w[i_in])
            
            # We need two items from out_set that fit in space_avail
            # and have combined value > v[i_in]
            
            # To speed up, we can sort out_list by value descending? 
            # Or just iterate. N=100. Pairs = ~5000.
            # Total ops per pass ~ 100 * 5000 = 500,000. 
            # 20 passes = 10M ops. Might be tight in Python if not optimized.
            # Let's limit the search space for 1-for-2.
            
            # Only consider top K out items by value?
            # Or just break early if we find one.
            
            # Optimization: If the sum of top 2 values in out_set is <= v[i_in], skip i_in?
            # Top 2 values in out_set:
            if len(out_list) < 2:
                continue
                
            # Get top 2 by value from out_list
            # out_list is sorted by density. Let's just take first few.
            # Actually, let's just do a limited pair check.
            
            # To keep it fast, we only check the first 20 items in out_list (best density)
            # for 1-for-2 swaps.
            candidate_outs = out_list[:20]
            
            k = len(candidate_outs)
            for a_idx in range(k):
                i_out1 = candidate_outs[a_idx]
                if w[i_out1] > space_avail:
                    continue # Too heavy, and others might be lighter? No, sorted by density not weight.
                    # Actually, if sorted by density, weight is random.
                    # We should probably sort candidate_outs by weight for this?
                    # Let's just check all pairs in the subset.
                
                for b_idx in range(a_idx + 1, k):
                    i_out2 = candidate_outs[b_idx]
                    
                    if w[i_out1] + w[i_out2] <= space_avail:
                        if v[i_out1] + v[i_out2] > v[i_in] + 1e-9:
                            # Accept
                            sol_set.remove(i_in)
                            sol_set.add(i_out1)
                            sol_set.add(i_out2)
                            
                            out_set.remove(i_out1)
                            out_set.remove(i_out2)
                            out_set.add(i_in)
                            
                            current_weight -= w[i_in]
                            current_weight += w[i_out1] + w[i_out2]
                            current_value -= v[i_in]
                            current_value += v[i_out1] + v[i_out2]
                            
                            improved = True
                            found_12 = True
                            break
                if found_12:
                    break
            if found_12:
                break
        
        if found_12:
            continue

        # --- 2-for-1 Swaps (Remove 2, Add 1) ---
        # Remove i_in1, i_in2, Add i_out.
        # Condition: w[i_out] - w[i_in1] - w[i_in2] <= C - current_weight + w[i_in1] + w[i_in2]
        #           => w[i_out] <= C - current_weight + w[i_in1] + w[i_in2]
        # Value Gain: v[i_out] - v[i_in1] - v[i_in2] > 0
        
        found_21 = False
        # Iterate out items (best density first)
        for i_out in out_list:
            if found_21:
                break
            
            # Space needed: w[i_out] - (C - current_weight)
            # We need to remove items with total weight >= w[i_out] - (C - current_weight)
            # Let deficit = w[i_out] - (C - current_weight)
            # If deficit <= 0, we can just add it? No, 2-for-1 implies removing 2.
            # If deficit <= 0, it fits in current space. That's a 0-for-1 add, which is handled by 1-for-1 if we consider removing nothing?
            # No, 1-for-1 is swap. If it fits, we should add it.
            # Wait, the greedy init fills space. If space remains, we can add items.
            # The local search above only does swaps (remove 1, add 1).
            # If there is space, we should add items!
            
            # Let's check if we can just ADD an item without removing anything.
            if w[i_out] <= C - current_weight:
                # Add it
                sol_set.add(i_out)
                out_set.remove(i_out)
                current_weight += w[i_out]
                current_value += v[i_out]
                improved = True
                found_21 = True # Using flag to break, though it's an add
                break
            
            # If it doesn't fit, we need to remove items.
            # We look for 2 items to remove.
            # To be efficient, we only check the worst 20 items in sol_set.
            candidate_ins = sol_list[:20]
            
            k = len(candidate_ins)
            for a_idx in range(k):
                i_in1 = candidate_ins[a_idx]
                if found_21:
                    break
                for b_idx in range(a_idx + 1, k):
                    i_in2 = candidate_ins[b_idx]
                    
                    # Check weight constraint
                    if w[i_out] <= (C - current_weight) + w[i_in1] + w[i_in2]:
                        # Check value constraint
                        if v[i_out] > v[i_in1] + v[i_in2] + 1e-9:
                            # Accept
                            sol_set.remove(i_in1)
                            sol_set.remove(i_in2)
                            sol_set.add(i_out)
                            
                            out_set.remove(i_out)
                            out_set.add(i_in1)
                            out_set.add(i_in2)
                            
                            current_weight -= w[i_in1] + w[i_in2]
                            current_weight += w[i_out]
                            current_value -= v[i_in1] + v[i_in2]
                            current_value += v[i_out]
                            
                            improved = True
                            found_21 = True
                            break
            if found_21:
                break
        
        if found_21:
            continue
        
        # No improvement in this pass
        break

    # Select an item from the best solution found
    if not sol_set:
        return -1
        
    # To ensure we don't get stuck in a loop or pick a bad item, 
    # pick the item with highest density from the current solution
    # This ensures we always return a valid index from the current feasible set
    # Note: The caller adds this item to their own knapsack.
    # Our "solution" is the set of items we *would* have picked if we were running the whole algorithm.
    # But the interface is "select NEXT item".
    # The prompt says: "picks the next item to add... while building 0/1 knapsack solutions".
    # This implies the state is passed in: remaining_capacity, remaining_weights, remaining_values.
    # My local search simulates the *rest* of the process from the current state.
    # So the "best solution" is the set of items I would pick from the *remaining* items.
    # I should return the first item of that optimal sequence.
    
    # How to determine the "first" item?
    # The greedy/local search built a set. The order of addition matters for the "next" item.
    # Usually, the item that was added first in the greedy phase or has highest priority in the final set is a good candidate.
    # However, the local search treats the set as a whole.
    # A safe heuristic: Return the item in the optimal set with the highest density.
    # This is consistent with the greedy logic.
    
    sol_arr = np.array(list(sol_set))
    sol_densities = densities[sol_arr]
    best_pick_rel = np.argmax(sol_densities)
    return int(sol_arr[best_pick_rel])
