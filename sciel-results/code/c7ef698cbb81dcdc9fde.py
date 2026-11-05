import numpy as np
import time
import sys

_start_time = time.time()
TIME_LIMIT = 110.0 # Leave 10s margin

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Time check
    if time.time() - _start_time > TIME_LIMIT:
        return -1
        
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
        
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    # Identify feasible items
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
        
    # Extract feasible items
    f_weights = weights[feasible_indices]
    f_values = values[feasible_indices]
    n_f = len(f_weights)
    
    # Determine scaling factor based on precision of weights
    # Convert to string to check decimal places, or use a heuristic
    # Heuristic: Check if multiplying by 10, 100, 1000, 10000 makes them integers
    scale = 1
    for s in [10, 100, 1000, 10000]:
        # Check if all weights * s are close to integers
        w_scaled = f_weights * s
        if np.allclose(w_scaled, np.round(w_scaled), atol=1e-9):
            scale = s
            break
            
    # Also check capacity
    cap_scaled = int(round(remaining_capacity * scale))
    
    # If cap is 0, return -1
    if cap_scaled <= 0:
        return -1
        
    # Weights scaled
    w_int = np.round(f_weights * scale).astype(int)
    
    # Filter out zero weight items? If weight is 0, value > 0, we should take it.
    # DP handles weight 0 correctly if we are careful.
    # If w=0, dp[i][c] = max(dp[i-1][c], dp[i-1][c] + v) = dp[i-1][c] + v.
    
    # DP Table
    # dp[i][c] = max value using first i items (0 to i-1) with capacity c
    # Dimensions: (n_f + 1) x (cap_scaled + 1)
    
    # Check if table is too large to allocate (safety)
    # 100 * 125000 = 12.5M floats ~ 100MB. 
    # If scale was 10000, cap=125000.
    # If n_f is 100.
    # This is acceptable.
    
    dp = np.zeros((n_f + 1, cap_scaled + 1), dtype=np.float64)
    # choice[i][c] = 1 if item i-1 is taken, 0 otherwise
    choice = np.zeros((n_f + 1, cap_scaled + 1), dtype=np.uint8)
    
    # Fill DP table
    for i in range(1, n_f + 1):
        w_i = w_int[i-1]
        v_i = f_values[i-1]
        
        # dp[i] = max(dp[i-1], dp[i-1][:-w_i] + v_i)
        # We need to be careful with indices.
        # For c from 0 to cap_scaled:
        # if c < w_i: dp[i][c] = dp[i-1][c]
        # else: dp[i][c] = max(dp[i-1][c], dp[i-1][c-w_i] + v_i)
        
        prev_dp = dp[i-1]
        curr_dp = dp[i]
        
        # Copy base case (not taking item)
        curr_dp[:] = prev_dp
        
        if w_i <= cap_scaled:
            # Taking item
            # c ranges from w_i to cap_scaled
            # curr_dp[w_i:] = np.maximum(curr_dp[w_i:], prev_dp[:-w_i] + v_i)
            
            # We need to track choice
            # If we take, choice[i][c] = 1
            # If we don't, choice[i][c] = 0
            
            take_vals = prev_dp[:-w_i] + v_i
            not_take_vals = curr_dp[w_i:] # which is prev_dp[w_i:]
            
            # Determine where taking is better
            # take_vals > not_take_vals
            take_mask = take_vals > not_take_vals
            
            # Update values
            curr_dp[w_i:][take_mask] = take_vals[take_mask]
            
            # Update choices
            # choice[i][w_i + idx] = 1 where take_mask[idx] is True
            # Indices in curr_dp[w_i:] correspond to w_i ... cap_scaled
            # So indices in choice[i] are w_i ... cap_scaled
            choice[i][w_i:][take_mask] = 1
            
    # Check if optimal value is 0 (or close to 0)
    if dp[n_f][cap_scaled] <= 1e-9:
        return -1
        
    # Reconstruct to find one optimal item
    c = cap_scaled
    for i in range(n_f, 0, -1):
        if choice[i][c] == 1:
            # Item i-1 is taken
            item_idx_in_feasible = i - 1
            # Map back to original index
            original_idx = feasible_indices[item_idx_in_feasible]
            return int(original_idx)
        else:
            # Item i-1 is not taken, move to previous item
            # Capacity remains same
            pass
            
    # If we reach here, it means no item was marked as taken in the trace?
    # This shouldn't happen if dp[n_f][cap] > 0.
    # However, if all values are 0, dp is 0. We handled that.
    # If there's a bug, fallback to greedy?
    
    # Fallback: Pick the feasible item with highest value/weight ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(f_weights > 0, f_values / f_weights, 0)
    best_f_idx = np.argmax(densities)
    return int(feasible_indices[best_f_idx])
