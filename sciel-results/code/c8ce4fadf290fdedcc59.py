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
    
    # Sort by density
    # Handle w=0
    keys = np.zeros(n)
    zero_w_pos_v = (w == 0) & (v > 0)
    keys[zero_w_pos_v] = 1e10
    pos_w = w > 0
    keys[pos_w] = v[pos_w] / w[pos_w]
    
    # Sort feasible indices by keys descending
    sorted_idx = feasible_indices[np.argsort(-keys[feasible_indices], kind='stable')]
    
    # Exact DP for completion value
    # Capacity is small (12.5), so we can scale it to integers.
    # Let's use a factor of 100 to be safe.
    scale = 100
    C_int = int(C * scale)
    
    # dp[c] = max value achievable with capacity c
    # We only need to consider feasible items.
    # Note: dp[0..C_int]
    # Initialize with 0 (assuming empty set is valid)
    dp = np.zeros(C_int + 1, dtype=np.float64)
    
    # We iterate through all feasible items to build the DP table.
    # This gives us the optimal value for the remaining items with capacity C_int.
    # But we need to evaluate the "completion" value for each candidate c.
    # Completion value for c = v[c] + optimal_value(remaining_items \ {c}, capacity C - w[c])
    
    # To get optimal_value(remaining_items \ {c}, capacity C - w[c]),
    # we can precompute the DP for all items, but that includes c.
    # If we precompute DP for all feasible items, let's call it DP_all.
    # DP_all[k] is the max value using a subset of feasible items with capacity k.
    # This doesn't directly give us the max value excluding c.
    
    # However, for 0/1 knapsack, if we want to exclude c, we can't just subtract.
    # But we can do a "forward" DP and a "backward" DP?
    # Or simply, since N is small (100), we can just run the DP for each candidate?
    # No, that's O(N * N * C) = 100 * 100 * 1250 = 12.5M ops.
    # 12.5M ops in Python/NumPy is fast (< 0.1s).
    # So we can just run a DP for each candidate c to find the best completion.
    
    best_val = -1e10
    best_idx = -1
    
    # Optimization: We can limit candidates to top K by density?
    # The current code checks all. Let's stick to all for correctness first, then optimize if needed.
    # But running DP 100 times might be slow if the constant is high.
    # Let's try running DP once for all items, and see if we can use it.
    # Actually, let's just run the DP for the specific set of items (all feasible) once.
    # Then for each c, we want max value from items \ {c}.
    # This is tricky with a single DP table.
    
    # Alternative:
    # Run DP for all feasible items.
    # dp_all[cap] = max value using subset of feasible items with total weight <= cap.
    # This is the optimal value for the remaining problem.
    # If we pick item c, the remaining capacity is C - w[c].
    # The remaining items are feasible_indices \ {c}.
    # We need max value from feasible_indices \ {c} with capacity C - w[c].
    
    # If we assume that the optimal solution for capacity C - w[c] using all feasible items
    # does NOT include c, then dp_all[C - w[c]] is the answer.
    # If it DOES include c, then we are double counting c? No, dp_all is for the set of items.
    # If dp_all uses c, then it's not a valid completion for "remaining items after picking c".
    
    # So we need to be careful.
    # Given the small N and C, let's just run a DP for each candidate c.
    # It's O(N^2 * C).
    # 100 * 100 * 1250 = 12.5M operations.
    # NumPy can handle this quickly.
    
    for c in sorted_idx:
        w_c = w[c]
        v_c = v[c]
        
        if w_c > C:
            continue
            
        rem_cap = C - w_c
        if rem_cap < 0:
            continue
            
        # Build DP for items excluding c
        # We can use the pre-sorted list or just iterate through feasible_indices
        # Let's use the original indices to be safe.
        
        # Initialize dp for this candidate
        # We only need capacity up to rem_cap
        rem_cap_int = int(rem_cap * scale)
        
        # If rem_cap is 0, dp is 0
        if rem_cap_int == 0:
            # Only items with weight 0 can be added
            # Sum of values of all feasible items with weight 0 (excluding c if c has weight 0)
            # But wait, if w_c > 0, then c is not in the weight 0 set.
            # If w_c == 0, we exclude c.
            
            # Let's just run the DP loop, it handles this.
            pass
            
        dp = np.zeros(rem_cap_int + 1, dtype=np.float64)
        
        # Iterate through all feasible items except c
        for idx in feasible_indices:
            if idx == c:
                continue
            w_i = w[idx]
            v_i = v[idx]
            
            if w_i > rem_cap:
                continue
                
            w_i_int = int(w_i * scale)
            
            if w_i_int == 0:
                # If weight is 0, we can always add it.
                # dp[k] += v_i for all k?
                # No, 0/1 knapsack. If w=0, we can pick it or not.
                # Since v_i >= 0 (assumed, or we can check), we should pick it if v_i > 0.
                # If v_i > 0, add to all dp[k].
                if v_i > 0:
                    dp += v_i
            else:
                # Standard 0/1 knapsack update
                # dp_new[k] = max(dp_old[k], dp_old[k - w_i] + v_i)
                # We can do this in-place if we iterate backwards.
                # But NumPy vectorized backwards is tricky.
                # For small C (1250), a Python loop is fine.
                # Or use np.maximum(dp, np.roll(dp, w_i_int) + v_i) ... no, roll wraps around.
                
                # Let's use a simple loop for the DP update for this item.
                # Or vectorized:
                # temp = dp.copy()
                # dp[w_i_int:] = np.maximum(dp[w_i_int:], dp[:-w_i_int] + v_i)
                
                # This is efficient.
                dp[w_i_int:] = np.maximum(dp[w_i_int:], dp[:-w_i_int] + v_i)
                
        completion_val = dp[rem_cap_int]
        total_val = v_c + completion_val
        
        if total_val > best_val:
            best_val = total_val
            best_idx = c

    if best_val <= 0:
        return -1
        
    return int(best_idx)
