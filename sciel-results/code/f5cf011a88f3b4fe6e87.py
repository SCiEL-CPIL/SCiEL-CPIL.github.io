"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack using Dynamic Programming to find an item
    that is part of the optimal solution for the current state.
    """
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Global deadline guard
    if not hasattr(select_next_item, '_start_time'):
        select_next_item._start_time = time.time()
    
    elapsed = time.time() - select_next_item._start_time
    if elapsed > 140:  # Safety margin
        return -1

    # Filter feasible items
    # Note: The input arrays are "remaining" arrays, so indices 0..n-1 correspond to the remaining items.
    # We need to return the index into these remaining arrays.
    
    feasible_indices = []
    feasible_weights = []
    feasible_values = []
    
    for i in range(n):
        if remaining_weights[i] <= remaining_capacity:
            feasible_indices.append(i)
            feasible_weights.append(remaining_weights[i])
            feasible_values.append(remaining_values[i])
            
    m = len(feasible_indices)
    if m == 0:
        return -1
        
    if m == 1:
        return feasible_indices[0]

    # Scale weights and capacity to integers
    # Capacity is 12.5, so scaling by 100 gives 1250.
    scale = 100
    cap_scaled = int(round(remaining_capacity * scale))
    
    # If capacity is too large for DP (unlikely with 12.5), fall back to greedy
    if cap_scaled > 50000:
        # Fallback: density
        densities = [v / w for w, v in zip(feasible_weights, feasible_values)]
        best_idx = max(range(m), key=lambda i: densities[i])
        return feasible_indices[best_idx]

    # DP Table
    # dp[i][c] = max value using items from i to m-1 with capacity c
    # Dimensions: (m+1) x (cap_scaled + 1)
    # Initialize with 0
    dp = [[0] * (cap_scaled + 1) for _ in range(m + 1)]
    
    # Fill DP table backwards
    # i goes from m-1 down to 0
    for i in range(m - 1, -1, -1):
        w_i_scaled = int(round(feasible_weights[i] * scale))
        v_i = feasible_values[i]
        
        # If weight is 0, handle carefully to avoid division by zero or infinite loops in logic, 
        # but standard knapsack handles w=0 fine (take it always if v>0).
        # Here we just update the table.
        
        # Optimization: only iterate c from cap_scaled down to w_i_scaled?
        # No, dp[i][c] depends on dp[i+1][c] and dp[i+1][c-w_i].
        # We can iterate c from 0 to cap_scaled.
        
        row_i = dp[i]
        row_next = dp[i+1]
        
        for c in range(cap_scaled + 1):
            if c < w_i_scaled:
                row_i[c] = row_next[c]
            else:
                take_val = v_i + row_next[c - w_i_scaled]
                skip_val = row_next[c]
                if take_val > skip_val:
                    row_i[c] = take_val
                else:
                    row_i[c] = skip_val

    # Find an item that is part of the optimal solution
    # The optimal value is dp[0][cap_scaled]
    optimal_val = dp[0][cap_scaled]
    
    # Iterate through items to find one that satisfies the condition
    # We look for i such that v_i + dp[i+1][cap_scaled - w_i] == dp[i][cap_scaled]
    # And dp[i][cap_scaled] must be part of the path from dp[0][cap_scaled].
    # Actually, if we just check if v_i + dp[i+1][cap_scaled - w_i] == dp[0][cap_scaled], 
    # it ensures that picking i leads to the global optimum.
    
    # We want to pick the "first" such item in the feasible list? 
    # Any valid item works.
    
    for i in range(m):
        w_i_scaled = int(round(feasible_weights[i] * scale))
        if w_i_scaled <= cap_scaled:
            # Check if picking this item allows achieving the optimal value
            # The value if we pick i is v_i + optimal_value(remaining items after i, remaining capacity)
            # The optimal value for remaining items after i is dp[i+1][cap_scaled - w_i_scaled]
            val_if_pick = feasible_values[i] + dp[i+1][cap_scaled - w_i_scaled]
            
            if val_if_pick == optimal_val:
                return feasible_indices[i]
                
    # If no item found (should not happen), fallback to best density
    densities = [v / w for w, v in zip(feasible_weights, feasible_values)]
    best_idx = max(range(m), key=lambda i: densities[i])
    return feasible_indices[best_idx]

# EVOLVE-BLOCK-END
