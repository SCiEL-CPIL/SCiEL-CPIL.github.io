import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 145.0 # Safety margin

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    if time.time() - _start_time > _TIMEOUT:
        return -1

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity
    
    if C < 0:
        return -1
        
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    # Lagrangian relaxation heuristic:
    # We want to find a lambda such that the total weight of items with v_i - lambda * w_i > 0 is close to C.
    # This is equivalent to finding the dual variable for the capacity constraint.
    # We can use binary search on lambda.
    
    # For w=0 items, they are always beneficial if v>0.
    # Let's separate them.
    zero_w_mask = (w == 0) & (v > 0)
    zero_w_indices = feasible_indices[zero_w_mask[feasible_indices]]
    
    pos_w_mask = (w > 0)
    pos_w_indices = feasible_indices[pos_w_mask[feasible_indices]]
    
    if len(pos_w_indices) == 0:
        # Only zero weight items
        if len(zero_w_indices) > 0:
            return int(zero_w_indices[0])
        return -1
        
    w_pos = w[pos_w_indices]
    v_pos = v[pos_w_indices]
    
    # Binary search for lambda
    # Lower bound: 0
    # Upper bound: max(v/w)
    lo = 0.0
    hi = np.max(v_pos / w_pos)
    
    # We want to find lambda such that sum(w_i for v_i > lambda * w_i) is close to C
    # Let's define a function that computes total weight for a given lambda
    def total_weight_for_lambda(lam):
        # Items where v_i > lam * w_i
        # v_i / w_i > lam
        densities = v_pos / w_pos
        selected = densities > lam
        return np.sum(w_pos[selected])
    
    # Binary search
    # If total_weight_for_lambda(lo) < C, we need to lower lambda (include more items)
    # If total_weight_for_lambda(hi) > C, we need to raise lambda (include fewer items)
    # Wait, if lam increases, fewer items are selected, so total weight decreases.
    # So if weight > C, we need to increase lam.
    # If weight < C, we need to decrease lam.
    
    # We want to find the lambda where weight crosses C.
    # Let's do 20 iterations of binary search.
    for _ in range(20):
        mid = (lo + hi) / 2.0
        tw = total_weight_for_lambda(mid)
        if tw > C:
            lo = mid
        else:
            hi = mid
            
    lambda_star = (lo + hi) / 2.0
    
    # Now, score each candidate item c in pos_w_indices:
    # Score = v_c + (value of optimal completion with lambda_star)
    # The "value of optimal completion" can be approximated by the Lagrangian dual value?
    # Or we can just use the Lagrangian score for the remaining items?
    # A better approach:
    # For each candidate c, the remaining capacity is C - w_c.
    # We can re-compute a new lambda for the remaining problem? That's expensive.
    # Instead, let's use the fixed lambda_star to score the completion.
    # Completion value approx = sum(v_i for i in remaining, v_i > lambda_star * w_i) ?
    # This is the greedy fill value using the threshold lambda_star.
    
    # Let's refine:
    # For each candidate c:
    # 1. Pick c.
    # 2. Remaining items: pos_w_indices \ {c}
    # 3. Remaining capacity: C - w_c
    # 4. Greedily fill remaining capacity with items from remaining items, sorted by density,
    #    but stop when total weight exceeds remaining capacity?
    #    Or use the lambda_star threshold?
    #    The lambda_star threshold selects items with density > lambda_star.
    #    This might exceed capacity or fall short.
    #    If it exceeds, we might need to adjust.
    #    But for a heuristic, let's just take the sum of values of items with density > lambda_star
    #    that fit in the remaining capacity? That's a knapsack problem again.
    
    # Simpler heuristic:
    # Score(c) = v_c + sum(v_i for i in remaining, density_i > lambda_star, w_i <= C - w_c)
    # This is not quite right because we need to respect the total weight constraint of the remaining items.
    # But it's a good approximation.
    
    # Let's try this:
    # Pre-compute the list of items sorted by density descending.
    densities = v_pos / w_pos
    sorted_pos_indices = pos_w_indices[np.argsort(-densities, kind='stable')]
    
    best_val = -1e10
    best_idx = -1
    
    # Also consider zero weight items separately?
    # If we pick a zero weight item, it doesn't consume capacity.
    # So we should always pick the best zero weight item first?
    # Or include them in the completion value.
    
    # Let's include zero weight items in the completion value for all candidates.
    # Since they don't consume capacity, they are always added if v > 0.
    zero_w_value_sum = np.sum(v[zero_w_indices]) if len(zero_w_indices) > 0 else 0.0
    
    for c in sorted_pos_indices:
        w_c = w[c]
        v_c = v[c]
        
        rem_cap = C - w_c
        if rem_cap < 0:
            continue
            
        # Compute completion value using lambda_star threshold
        # Items with density > lambda_star
        # And weight <= rem_cap (individually feasible)
        # But we need to respect the total weight constraint.
        # Let's just sum the values of items with density > lambda_star that fit individually.
        # This is an upper bound, but a good heuristic.
        
        # To be more accurate, we can do a quick greedy fill for the top candidates only.
        # But let's stick to the simple threshold sum first.
        
        # Indices of items with density > lambda_star, excluding c
        # We can pre-compute a boolean mask for density > lambda_star
        # But c changes, so we need to be careful.
        
        # Let's just iterate through the sorted list and sum values until weight exceeds rem_cap?
        # No, that's O(N) per candidate, so O(N^2) total. 100*100 = 10,000. Fast.
        
        # Let's do a greedy fill for the completion.
        # This is more accurate than just summing all items with density > lambda_star.
        
        current_weight = 0
        current_value = 0
        
        for idx in sorted_pos_indices:
            if idx == c:
                continue
            w_i = w[idx]
            v_i = v[idx]
            
            if w_i > rem_cap - current_weight:
                continue
                
            # Check if density is above some threshold?
            # The lambda_star heuristic suggests we should only pick items with density > lambda_star.
            # But if we have capacity left, we might pick items with lower density?
            # In a greedy fill, we pick the highest density items that fit.
            # So let's just do a standard greedy fill.
            
            current_weight += w_i
            current_value += v_i
            
            if current_weight >= rem_cap:
                break
                
        completion_val = current_value + zero_w_value_sum
        total_val = v_c + completion_val
        
        if total_val > best_val:
            best_val = total_val
            best_idx = c

    # Also consider picking a zero weight item first?
    # If we pick a zero weight item, the completion value is the greedy fill of all pos_w items with capacity C.
    # And we add the value of the zero weight item.
    # But we can only pick one item per call.
    # So if we pick a zero weight item, the total value is v_c + greedy_fill(C).
    # Let's compute greedy_fill(C) once.
    
    current_weight = 0
    current_value = 0
    for idx in sorted_pos_indices:
        w_i = w[idx]
        v_i = v[idx]
        if w_i > C - current_weight:
            continue
        current_weight += w_i
        current_value += v_i
        if current_weight >= C:
            break
            
    greedy_fill_C = current_value + zero_w_value_sum
    
    # Now check zero weight items
    for c in zero_w_indices:
        v_c = v[c]
        # Completion value is greedy_fill(C) minus the value of c if it was included in greedy_fill?
        # No, greedy_fill only includes pos_w items.
        # So completion value is greedy_fill(C).
        # But wait, if we pick c, the remaining items are all pos_w items and other zero_w items.
        # The greedy_fill(C) already includes all pos_w items that fit.
        # It does not include other zero_w items.
        # So we should add the sum of values of other zero_w items.
        
        other_zero_w_value = zero_w_value_sum - v_c
        
        total_val = v_c + greedy_fill_C + other_zero_w_value
        
        if total_val > best_val:
            best_val = total_val
            best_idx = c

    if best_val <= 0:
        return -1
        
    return int(best_idx)
