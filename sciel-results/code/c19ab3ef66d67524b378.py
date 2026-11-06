import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 110.0  # Safety margin before 120s/150s kill

def _greedy_pick(remaining_weights, remaining_values):
    """Fallback greedy strategy: highest value density."""
    if len(remaining_weights) == 0:
        return -1
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = remaining_values / remaining_weights
    # If any density is inf or nan, handle it
    # If weight is 0, value should be handled. If w=0, v>0, density inf.
    # np.argmax handles inf correctly (returns first inf).
    # If all are nan? unlikely.
    max_idx = int(np.argmax(densities))
    return max_idx

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack using DP approximation.
    Returns the index into the remaining arrays, or -1 to stop.
    """
    current_time = time.time()
    elapsed = current_time - _START_TIME
    
    # Fallback to greedy if running out of time
    if elapsed > _TIME_LIMIT:
        return _greedy_pick(remaining_weights, remaining_values)
        
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # Scale capacity and weights to integers
    # Capacity is 12.5. Scale by 1000 -> 12500.
    scale = 1000
    cap_int = int(remaining_capacity * scale)
    
    # Filter feasible items
    weights_arr = np.array(remaining_weights, dtype=float)
    values_arr = np.array(remaining_values, dtype=float)
    
    # Ensure weights are not negative (problem implies non-negative)
    # Filter items that fit
    feasible_mask = weights_arr <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights_arr[feasible_mask]
    feasible_values = values_arr[feasible_mask]
    
    m = len(feasible_weights)
    
    # If only one item, return it
    if m == 1:
        return int(feasible_indices[0])
        
    # Convert weights to int
    fw_int = np.floor(feasible_weights * scale).astype(int)
    
    # Check if any weight is 0
    # If weight is 0, it's always beneficial to take if value > 0.
    # DP handles w=0 correctly?
    # dp[:-0] is dp[:]. dp[0:] is dp[0:].
    # val_take = dp + v.
    # val_skip = dp.
    # take = dp + v > dp -> v > 0.
    # So yes, it works.
    
    # DP Initialization
    # dp[c] = max value using processed items with capacity c
    dp = np.zeros(cap_int + 1, dtype=float)
    takes = []
    
    # Process each feasible item
    for i in range(m):
        w = fw_int[i]
        v = feasible_values[i]
        
        if w > cap_int:
            # Should not happen due to filter, but just in case
            takes.append(np.zeros(cap_int + 1, dtype=bool))
            continue
            
        if w == 0:
            # If weight is 0, we take it if value > 0
            if v > 0:
                dp = dp + v
                takes.append(np.ones(cap_int + 1, dtype=bool))
            else:
                takes.append(np.zeros(cap_int + 1, dtype=bool))
            continue
            
        # Vectorized DP update
        # val_take: capacities w..cap_int get dp[0..cap_int-w] + v
        # val_skip: capacities w..cap_int get dp[w..cap_int]
        
        # Slice: dp[:-w] is 0..cap_int-w. Length cap_int-w+1.
        # Slice: dp[w:] is w..cap_int. Length cap_int-w+1.
        
        val_take = dp[:-w] + v
        val_skip = dp[w:]
        
        # Determine which is better
        # take_mask for capacities w..cap_int
        take_mask_segment = val_take > val_skip
        
        # Construct full take array
        take_arr = np.zeros(cap_int + 1, dtype=bool)
        take_arr[w:] = take_mask_segment
        
        # Update DP values
        # new_dp[w:] = max(val_skip, val_take)
        # new_dp[:w] = dp[:w] (unchanged)
        
        dp[w:] = np.maximum(val_skip, val_take)
        
        takes.append(take_arr)
        
        # Periodic time check
        if (i & 15) == 0:
            if time.time() - current_time > 5.0:
                # If this single DP step is taking too long (unlikely for small N), fallback
                return _greedy_pick(remaining_weights, remaining_values)

    # Check optimal value
    if dp[cap_int] <= 0:
        return -1
        
    # Traceback
    c = cap_int
    for i in range(m - 1, -1, -1):
        if takes[i][c]:
            # Item i (in filtered list) is taken
            return int(feasible_indices[i])
        # If not taken, c remains the same for the previous item
        # Note: if we had taken it, we would have returned.
        # So we just continue to the previous item with same c.
        
    # If we reach here, something went wrong or no item was taken despite opt val > 0?
    # This can happen if opt val is 0, which we checked.
    # Or if traceback logic is flawed.
    # Fallback
    return _greedy_pick(remaining_weights, remaining_values)
# EVOLVE-BLOCK-END
