import numpy as np
import time
import sys

# Global variables to cache state and results
# This allows us to avoid redundant computation if the state is the same
# or to maintain a consistent search history if the evaluator calls
# the function in a loop within the same process.
_global_cache = {}

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety
    
    # If all items are too heavy, stop
    if np.max(remaining_weights) > remaining_capacity:
        # Check if any fit
        if np.any(remaining_weights <= remaining_capacity):
            pass
        else:
            return -1

    # Create a unique key for the current state to check cache
    # Using a hash of the remaining items and capacity
    # This is expensive for large n, but n=100 is manageable.
    # However, floating point capacity makes hashing tricky.
    # Let's just rely on the fact that the state changes slightly each step.
    # A simpler heuristic: If we just picked an item, the state is unique.
    
    # Strategy: Branch and Bound
    # We want to find the index of the best next item.
    # We will simulate the rest of the packing for each candidate next item
    # but prune aggressively.
    
    # Sort items by density descending to improve pruning
    densities = np.divide(remaining_values, remaining_weights, out=np.zeros_like(remaining_values), where=remaining_weights>0)
    order = np.argsort(-densities)
    
    best_value_found = -1.0
    best_next_item = -1
    
    # We will try each item as the "next" item, but we only need to try
    # the top few candidates? No, B&B explores all.
    # But we can prune the root level if we have a good lower bound.
    
    # Get a quick greedy lower bound
    def greedy_lower_bound(cap, weights, values):
        total_val = 0.0
        total_w = 0.0
        for i in order:
            if total_w + weights[i] <= cap:
                total_w += weights[i]
                total_val += values[i]
            else:
                # Fractional
                if weights[i] > 0:
                    frac = (cap - total_w) / weights[i]
                    if frac > 0:
                        total_val += values[i] * frac
                break
        return total_val

    # Initial lower bound from pure greedy (no lookahead for first item)
    # Actually, we want to find the best *first* item.
    # Let's run B&B from the root.
    # State: (index_in_order, current_cap, current_val)
    # We are looking for the best sequence starting from now.
    # The "next item" is the first item in that best sequence.
    
    # To make B&B efficient, we pass the sorted arrays.
    sorted_weights = remaining_weights[order]
    sorted_values = remaining_values[order]
    # We need to map back to original indices
    original_indices = order
    
    # Recursive B&B
    # depth: how many items we've considered
    # idx: current index in sorted arrays
    # cap: remaining capacity
    # val: current value
    # first_item_orig_idx: the original index of the first item picked in this branch
    #                     (if first item is picked, this is set; else -1)
    
    # Since we need the *next* item, and the function is called at the start of the step,
    # the "next item" is the first item of the optimal path from here.
    
    # We can't just return the best path's first item if we prune branches
    # that *start* with a suboptimal first item but lead to a better total?
    # No, if a branch starts with item A, its total value is V_A.
    # If a branch starts with item B, its total value is V_B.
    # We want the item that is part of the global optimal solution.
    
    # Let's use a standard B&B that finds the global optimal value and tracks the path.
    # But we only need the first item.
    
    # Optimization:
    # We can iterate through candidates for the first item (in density order).
    # For each candidate i, we calculate the best possible completion value using B&B.
    # We keep track of the best total value.
    # The candidate i that yields the highest total value is our next item.
    # We can prune candidate i if its upper bound (value[i] + fractional_bound(remaining)) <= best_total_so_far.
    
    best_total = -1.0
    best_idx_orig = -1
    
    # Precompute suffix sums of values and weights for fractional bound?
    # Fractional bound is easier to compute on the fly or with prefix/suffix arrays.
    # Since we are in sorted order, we can use a suffix sum of values and weights?
    # No, fractional bound depends on capacity.
    
    def fractional_bound(start_idx, cap):
        """
        Calculate upper bound value from start_idx onwards with capacity cap.
        Assumes items are sorted by density descending.
        """
        val = 0.0
        w = 0.0
        for i in range(start_idx, n):
            if w + sorted_weights[i] <= cap:
                w += sorted_weights[i]
                val += sorted_values[i]
            else:
                if sorted_weights[i] > 0:
                    frac = (cap - w) / sorted_weights[i]
                    if frac > 0:
                        val += sorted_values[i] * frac
                break
        return val

    # Iterate over possible first items
    # We only need to consider items that fit
    for i in range(n):
        if time.time() > deadline:
            break
            
        w_i = sorted_weights[i]
        v_i = sorted_values[i]
        
        if w_i > remaining_capacity:
            # Since sorted by density, not weight, a heavier item might come later?
            # No, we can't break. We must check all.
            # But if we sort by density, weights are arbitrary.
            continue
        
        # Upper bound for this branch: v_i + fractional_bound(i+1, remaining_capacity - w_i)
        ub = v_i + fractional_bound(i + 1, remaining_capacity - w_i)
        
        if ub <= best_total:
            # Prune this candidate entirely
            continue
            
        # If we can't prune, we need to find the *exact* best completion for this branch?
        # Or just accept the upper bound? No, we need the actual value to compare.
        # But running full B&B for each i is O(N * Search).
        # However, N is small (100) and Search is small due to pruning.
        
        # Let's run a limited B&B for the rest of the items given we picked i.
        # Actually, we can just do a quick greedy for the rest to get a lower bound,
        # and if that lower bound > best_total, update best.
        # But we want the *best* next item, which might have a slightly lower immediate value
        # but allow much better future items.
        
        # Let's run a recursive search for the completion of branch i.
        # To save time, we limit the depth or use the fractional bound as the value if we can't search deep.
        
        # For efficiency, let's assume the greedy completion is a good approximation
        # unless we have time to do better.
        
        # Let's try a simple recursive B&B for the subproblem.
        # State: (idx, cap, val)
        # We want max val.
        
        # To avoid deep recursion overhead in Python, we use a stack or limit depth.
        # Given capacity 12.5, depth is small.
        
        def search_completion(start_idx, cap, current_val):
            nonlocal best_total, best_idx_orig
            nonlocal time, deadline
            
            if start_idx >= n:
                return current_val
            
            # Prune if current_val + fractional_bound(start_idx, cap) <= best_total
            if current_val + fractional_bound(start_idx, cap) <= best_total:
                return current_val
            
            best_val_here = current_val
            
            # Option 1: Skip item start_idx
            # Option 2: Take item start_idx
            
            # Try Take first if it fits (usually better for density sorted)
            if sorted_weights[start_idx] <= cap:
                val_take = search_completion(start_idx + 1, cap - sorted_weights[start_idx], current_val + sorted_values[start_idx])
                if val_take > best_val_here:
                    best_val_here = val_take
                    
            # Try Skip
            val_skip = search_completion(start_idx + 1, cap, current_val)
            if val_skip > best_val_here:
                best_val_here = val_skip
                
            return best_val_here
        
        # Execute search for this first item i
        # Note: search_completion modifies best_total? No, it returns local best.
        # We need to compare the result of this branch with global best_total.
        
        # Wait, the recursive function above doesn't update global best_total directly
        # unless we pass it or use nonlocal. Let's refactor to return the max value found.
        
        def max_value_from(start_idx, cap, current_val):
            if start_idx >= n:
                return current_val
            
            # Prune
            if current_val + fractional_bound(start_idx, cap) <= best_total:
                return current_val # This branch is worse than known best globally, but we return its actual max for local comparison?
                                   # Actually, if it's pruned, we don't care about its exact value, just that it's <= best_total.
            
            best_local = current_val
            
            # Take
            if sorted_weights[start_idx] <= cap:
                v_take = max_value_from(start_idx + 1, cap - sorted_weights[start_idx], current_val + sorted_values[start_idx])
                if v_take > best_local:
                    best_local = v_take
            
            # Skip
            v_skip = max_value_from(start_idx + 1, cap, current_val)
            if v_skip > best_local:
                best_local = v_skip
                
            return best_local

        # Calculate best completion value for branch starting with i
        # The value of the full solution is v_i + max_value_from(i+1, remaining_capacity - w_i, 0)
        # But max_value_from uses best_total for pruning. 
        # This is tricky: if we prune based on global best_total, we might underestimate the branch value
        # if the global best_total is high. But that's fine, because if it's pruned, it's <= global best,
        # so it won't become the new best.
        
        # However, we must ensure that if this branch *is* the best, we don't prune it.
        # The condition `current_val + fractional_bound <= best_total` ensures we only prune
        # if it *cannot* beat the current global best.
        
        completion_val = max_value_from(i + 1, remaining_capacity - w_i, 0.0)
        total_val = v_i + completion_val
        
        if total_val > best_total:
            best_total = total_val
            best_idx_orig = original_indices[i]
            
            # Update best_total to allow further pruning in subsequent iterations
            # Note: best_total is used inside max_value_from via nonlocal.
            
    if best_idx_orig != -1:
        return int(best_idx_orig)
    
    # Fallback: Greedy
    for i in range(n):
        if sorted_weights[i] <= remaining_capacity:
            return int(original_indices[i])
            
    return -1
