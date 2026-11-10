import numpy as np
import time
import sys

sys.setrecursionlimit(10000)

# Global deadline guard
# The evaluator kills after 150s. We want to finish with margin.
# Since select_next_item is called iteratively (up to 100 times),
# we should allocate a small budget per call or track global time.
# However, the function signature does not allow passing state.
# We can use a module-level variable to track start time.
_start_time = None
_GLOBAL_DEADLINE = 140.0  # Leave 10s buffer

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_time_up():
    return _get_elapsed() > _GLOBAL_DEADLINE

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if _is_time_up():
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # Handle zero weight positive value items first
    zero_weight_indices = np.where((weights == 0) & (values > 0))[0]
    if len(zero_weight_indices) > 0:
        return int(zero_weight_indices[0])
    
    # Filter feasible items (weight > 0 and weight <= capacity)
    feasible_mask = (weights <= remaining_capacity) & (weights > 0)
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
        
    # Strategy:
    # 1. Try DP if the problem size is small enough.
    # 2. If DP is too large, use Vectorized B&B.
    
    C = remaining_capacity
    max_w = np.max(f_weights)
    
    # DP Feasibility Check
    # We want to scale weights to integers.
    # Let scale be such that max_w * scale is manageable.
    # DP table size: f_n * (C * scale)
    # Memory limit: ~100MB -> 100e6 / 8 bytes = 12.5M booleans (for takes) + floats.
    # Let's aim for C_scaled * f_n < 5_000_000 operations.
    
    # Estimate scale
    # If we pick scale = 100, C_scaled = 1250. f_n=100. 125k ops. Very fast.
    # If we pick scale = 1000, C_scaled = 12500. f_n=100. 1.25M ops. Fast.
    # If we pick scale = 10000, C_scaled = 125000. f_n=100. 12.5M ops. ~0.5s-1s.
    
    # Let's try to maximize precision (scale) while staying under time limit.
    # Time limit per call: ~1s. Total 100 calls -> 100s.
    # But B&B might be needed.
    
    # Let's set a target for C_scaled.
    # If C is small (e.g. 12.5), we can use high scale.
    # If C is large, lower scale.
    
    # Heuristic:
    # Max C_scaled = 200000 (200k).
    # If C * scale > 200000, reduce scale.
    # Also check f_n * C_scaled < 10_000_000.
    
    target_C_scaled = 100_000
    scale = target_C_scaled / C
    scale = int(scale)
    scale = max(1, scale)
    
    # Ensure weights are not zero after scaling?
    # If max_w * scale == 0, then weights are too small. Increase scale.
    if max_w * scale < 1:
         scale = int(1.0 / max_w) + 10
         
    w_scaled = (f_weights * scale).astype(np.int64)
    C_scaled = int(C * scale)
    
    # Check if DP is viable
    # Memory for takes: f_n * (C_scaled + 1) booleans.
    # If this is too big, use B&B.
    dp_size = f_n * (C_scaled + 1)
    
    if dp_size < 5_000_000 and C_scaled < 200_000:
        # Use DP
        return _solve_dp(f_values, w_scaled, C_scaled, feasible_indices, f_n)
    else:
        # Use B&B
        return _solve_bb(f_values, w_scaled, C_scaled, feasible_indices, f_n)

def _solve_dp(values, weights, capacity, original_indices, n):
    # DP with backtracking
    # dp[j] = max value using capacity j
    # takes[i][j] = True if item i was taken to achieve dp[j]
    
    # Optimization: Use 1D dp for values, but we need backtracking.
    # Storing full takes matrix is O(N*C).
    # Alternative: Store predecessor item for each (i, j)? No, 0/1 knapsack doesn't have unique predecessor easily without storing choices.
    # Since N is small (<=100) and C is bounded, storing bool matrix is fine.
    
    dp = np.zeros(capacity + 1, dtype=np.float64)
    takes = np.zeros((n, capacity + 1), dtype=bool)
    
    for i in range(n):
        w = weights[i]
        v = values[i]
        if w == 0:
            continue
        
        # Vectorized update
        # We need to update dp[j] for j from capacity down to w.
        # candidate = dp[j-w] + v
        # if candidate > dp[j], take item.
        
        # Note: dp[:-w] is dp[0..capacity-w-1]
        # dp[w:] is dp[w..capacity]
        
        # We must be careful with in-place updates.
        # dp[w:] = max(dp[w:], dp[:-w] + v)
        # This works because np.maximum evaluates inputs before assignment.
        
        candidate = dp[:-w] + v
        mask = candidate > dp[w:]
        
        if np.any(mask):
            # Update dp
            dp[w:][mask] = candidate[mask]
            # Record take
            takes[i][w:][mask] = True
            
    # Backtrack
    # Find best j. Since value is non-decreasing with capacity, best is at capacity.
    # However, due to scaling, maybe not all capacity is used.
    # dp[capacity] is the max value achievable with capacity <= capacity.
    
    j = capacity
    selected_f_indices = []
    
    for i in range(n - 1, -1, -1):
        if takes[i][j]:
            selected_f_indices.append(i)
            j -= weights[i]
            if j < 0:
                break
                
    if not selected_f_indices:
        # Fallback: Pick best single item that fits
        # This should rarely happen if DP is correct, but safe.
        # Find item with max value/weight or just max value?
        # Just pick the one with max value that fits.
        best_val = -1
        best_idx = -1
        for idx in range(n):
            if weights[idx] <= capacity:
                if values[idx] > best_val:
                    best_val = values[idx]
                    best_idx = idx
        if best_idx == -1:
            return -1
        return int(original_indices[best_idx])
        
    # Pick one item from the optimal set
    # Any item from the optimal set is a valid choice for the next step.
    # We return the index of the first item in the backtracking list (which corresponds to the largest item index in the sorted order? No, just one of them).
    f_idx_to_pick = selected_f_indices[0]
    orig_idx = original_indices[f_idx_to_pick]
    
    return int(orig_idx)

def _solve_bb(values, weights, capacity, original_indices, n):
    # Vectorized B&B
    # Sort by density (value/weight)
    densities = values / weights
    order = np.argsort(-densities)
    
    sorted_values = values[order]
    sorted_weights = weights[order]
    sorted_orig_indices = original_indices[order]
    
    # Precompute prefix sums for fractional bound
    prefix_values = np.cumsum(sorted_values)
    prefix_weights = np.cumsum(sorted_weights)
    
    best_val = 0.0
    best_set = set()
    
    # DFS with stack to avoid recursion limit and overhead?
    # Recursion is fine for depth 100.
    
    start_time = time.time()
    time_limit = 0.5 # 0.5s per call max
    
    def dfs(i, curr_val, curr_weight, curr_set):
        nonlocal best_val, best_set
        if time.time() - start_time > time_limit:
            return
        
        if i == n:
            if curr_val > best_val:
                best_val = curr_val
                best_set = set(curr_set)
            return
            
        # Fractional Bound
        # We have items i..n-1 remaining.
        # Current value: curr_val
        # Remaining capacity: capacity - curr_weight
        
        rem_cap = capacity - curr_weight
        if rem_cap <= 0:
            if curr_val > best_val:
                best_val = curr_val
                best_set = set(curr_set)
            return
            
        # Calculate fractional bound for items i..n-1
        # If sum of remaining weights <= rem_cap, take all.
        # Else, take all until full, then fraction of next.
        
        # Use prefix sums to get sum of values/weights from i to end
        # Sum values i..n-1 = prefix_values[-1] - (prefix_values[i-1] if i>0 else 0)
        
        sum_rem_values = prefix_values[-1] - (prefix_values[i-1] if i > 0 else 0)
        sum_rem_weights = prefix_weights[-1] - (prefix_weights[i-1] if i > 0 else 0)
        
        if sum_rem_weights <= rem_cap:
            ub = curr_val + sum_rem_values
        else:
            # Need to find the item that breaks the capacity
            # We want the largest k >= i such that sum(weights[i..k-1]) <= rem_cap
            # Let prefix_w_rel = prefix_weights - (prefix_weights[i-1] if i>0 else 0)
            # We need prefix_w_rel[k-1] <= rem_cap < prefix_w_rel[k]
            
            # Vectorized search?
            # Create array of remaining weights starting from i
            rem_w = sorted_weights[i:]
            rem_v = sorted_values[i:]
            
            # Cumulative sum of rem_w
            cum_w = np.cumsum(rem_w)
            
            # Find where cum_w > rem_cap
            # np.searchsorted(cum_w, rem_cap, side='right') gives index of first element > rem_cap
            # If all <= rem_cap, we handled above.
            # So there is at least one > rem_cap.
            
            # idx_in_rem is index in rem_w
            # We want the last index where cum_w <= rem_cap.
            # searchsorted(cum_w, rem_cap, side='right') - 1
            # Let k be that index.
            # Then we take all items 0..k in rem_w, and fraction of k+1.
            
            k = np.searchsorted(cum_w, rem_cap, side='right') - 1
            
            if k < 0:
                # First item is too heavy?
                # Fraction of first item
                ub = curr_val + rem_v[0] * (rem_cap / rem_w[0])
            else:
                # Take all items 0..k
                val_full = np.sum(rem_v[:k+1])
                # Fraction of item k+1
                if k+1 < len(rem_w):
                    frac_val = rem_v[k+1] * (rem_cap - cum_w[k]) / rem_w[k+1]
                else:
                    frac_val = 0.0
                ub = curr_val + val_full + frac_val
        
        if ub <= best_val:
            return
            
        # Branching: Include item i first (heuristic: include if it fits and is promising)
        # Actually, standard B&B:
        # 1. Try include item i
        # 2. Try exclude item i
        
        v_i = sorted_values[i]
        w_i = sorted_weights[i]
        idx_i = sorted_orig_indices[i]
        
        # Include
        if curr_weight + w_i <= capacity:
            curr_set.add(idx_i)
            dfs(i + 1, curr_val + v_i, curr_weight + w_i, curr_set)
            curr_set.remove(idx_i)
            
        # Exclude
        dfs(i + 1, curr_val, curr_weight, curr_set)
        
    curr_set = set()
    dfs(0, 0.0, 0.0, curr_set)
    
    if not best_set:
        # Fallback: Pick best single item
        # Find item with max value that fits
        feasible = sorted_weights <= capacity
        if not np.any(feasible):
            return -1
        best_idx_in_sorted = np.where(feasible)[0][0]
        # Actually pick max value among feasible
        vals_feasible = sorted_values[feasible]
        best_local = np.argmax(vals_feasible)
        best_global_idx = np.where(feasible)[0][best_local]
        return int(sorted_orig_indices[best_global_idx])
        
    # Pick one item from best_set
    # To be consistent with DP which picks from backtracking (often larger indices),
    # we can just pick any.
    # Let's pick the one with highest density? Or just the first one.
    # The caller removes the item. It doesn't matter which one we pick from the optimal set.
    # However, picking a "core" item might be better?
    # For now, pick the first one in the set.
    # To make it deterministic, sort the set?
    # Sets are unordered.
    # Let's pick the item with the highest density from the best_set.
    
    best_item_idx = None
    max_density = -1
    for idx in best_set:
        # Find original index in sorted array?
        # We have sorted_orig_indices.
        # We need to map idx (original index) back to sorted position?
        # Or just calculate density from original arrays?
        # We don't have original arrays here, only sorted.
        # Let's just pick the first one.
        best_item_idx = idx
        break
        
    return int(best_item_idx)
