import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 1e-12:
        return -1
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Convert to numpy arrays for convenience
    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # Filter feasible items
    feasible_mask = weights <= remaining_capacity + 1e-12
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # --- Strategy: Hybrid Exact DP (when feasible) + Fractional Lookahead (fallback) ---
    
    # 1. Attempt Exact DP if the problem size is small enough to be fast.
    # Capacity is 12.5. Scaling by 100 gives 1250.
    # N is up to 100. DP table size ~ 100 * 1250 = 125,000 ops. Very fast.
    # However, we only need to pick the *first* item.
    # We can run the full DP to find the optimal set, then pick the first item in that set?
    # No, the DP gives the optimal total value. To get the first item, we need to know which item was taken first in the optimal sequence?
    # Actually, the problem asks for a heuristic that picks the *next* item.
    # If we use DP, we can determine the *entire* optimal solution for the current state.
    # Then we can just return the first item of that optimal solution.
    # This is a "perfect" move for the current state.
    
    # Let's try to run a scaled integer DP.
    # Scale factor: 100 seems safe for 12.5 capacity.
    # Weights and values are floats. We must ensure they scale to integers properly.
    # The problem states "generated 100-item instances with capacity 12.5".
    # It doesn't explicitly say weights/values are 2-decimal places, but 100x scaling is a common heuristic for such small capacities.
    # If weights have more precision, rounding might cause issues.
    # Let's check if we can use a larger scale or if we should stick to the fractional heuristic.
    # The current program (score 1.0) uses fractional lookahead.
    # Inspiration 1 (score 1.001) uses DP.
    # Let's combine: Try DP first. If it finishes quickly, return the first item of the optimal solution.
    # If DP is too slow or fails (e.g., due to precision issues), fall back to the fractional lookahead heuristic.
    
    # Check time budget for DP
    # DP is O(N * C_scaled).
    # N=100, C_scaled=1250. Ops = 125,000. This is trivial for Python/NumPy if vectorized, but standard loops might take ~0.01-0.05s.
    # We have 110s. DP is very safe.
    
    # Potential issue: Precision.
    # If weights are like 1.2345, scaling by 100 gives 123.45 -> 123.
    # This changes the problem.
    # The task says "scored by mean total value on eight generated 100-item instances".
    # Usually, these instances have weights/values that are "nice" or the DP approach in Inspiration 1 worked (score 1.001).
    # Let's assume the DP approach is viable but add a robustness check.
    # If the max weight * scale > some large number, or if n is huge, skip DP.
    
    # Let's try a refined DP approach:
    # 1. Scale weights and values.
    # 2. Run DP to find max value.
    # 3. Backtrack to find the set of items.
    # 4. Return the first item in the set (or any item in the set? The order doesn't matter for the set, but the function returns an index. The evaluator builds the solution incrementally. If we return an item from the optimal set, and the evaluator removes it, the next call will have a new state. If we always return an item from the optimal set of the *current* state, we will converge to the optimal solution).
    
    # Wait, if we return an item from the optimal set of the *current* state, does it matter which one?
    # Yes, because removing different items leads to different sub-problems.
    # However, if we pick the item that is part of *an* optimal solution, we are on a path to an optimal solution.
    # Actually, we should pick the item that allows the *rest* of the solution to be optimal.
    # The DP backtracking gives a specific optimal set. Any item in that set is a valid "next" move that leads to an optimal total value (assuming the rest of the moves are also optimal).
    # So, we can just pick the first item found in the backtrack.
    
    # Let's implement the DP with a time guard.
    
    scale = 100
    cap_int = int(remaining_capacity * scale)
    
    if cap_int <= 0:
        return -1
    
    w_float = weights
    v_float = values
    
    # Check if scaling is safe (i.e., values don't blow up)
    max_w = np.max(w_float)
    max_v = np.max(v_float)
    
    # If weights are very small, scale might need to be larger?
    # But capacity is fixed at 12.5.
    # If weights are e.g. 0.001, scale 100 -> 0.1 -> 0. That's bad.
    # Let's check min weight.
    min_w = np.min(w_float[w_float > 0]) if np.any(w_float > 0) else 1.0
    
    # If min_w * scale < 1, we lose precision.
    # Let's try to auto-detect scale or just stick to 100 if it seems safe.
    # Given the inspiration 1 worked with scale 100, let's assume it's okay.
    # But to be safer, let's check if any weight becomes 0 after rounding.
    w_int = np.round(w_float * scale).astype(int)
    v_int = np.round(v_float * scale).astype(int)
    
    # If any positive weight became 0, DP is invalid.
    if np.any((w_float > 1e-9) & (w_int == 0)):
        # Fall back to fractional heuristic
        return _fractional_heuristic(remaining_capacity, weights, values, feasible_indices, start_time, deadline)
    
    # Also, if any weight is > capacity, it's infeasible, but we already filtered feasible_indices.
    # However, DP runs on ALL items? No, we should only run DP on feasible items?
    # Actually, infeasible items (w > cap) will never be taken in DP anyway.
    # But to save time, we can filter them out before DP.
    # The `feasible_indices` are the ones that fit.
    # Let's run DP only on feasible items.
    
    w_dp = w_int[feasible_indices]
    v_dp = v_int[feasible_indices]
    
    # If no feasible items in DP (shouldn't happen if feasible_indices is not empty)
    if len(w_dp) == 0:
        return -1
    
    n_dp = len(w_dp)
    
    # DP Table: (n_dp + 1) x (cap_int + 1)
    # Memory: 101 * 1251 * 8 bytes (float/int) ~ 1MB. Fine.
    # Time: 100 * 1250 = 125k iterations. Fast.
    
    dp_table = np.zeros((n_dp + 1, cap_int + 1), dtype=np.int64)
    
    # Fill DP table
    # We can vectorize the inner loop?
    # for i in range(1, n_dp + 1):
    #    wi = w_dp[i-1]
    #    vi = v_dp[i-1]
    #    # dp[i, :] = np.maximum(dp[i-1, :], dp[i-1, :wi] + vi)  <-- This is tricky with numpy slicing
    #    # Standard loop is probably fast enough for 125k ops.
    
    for i in range(1, n_dp + 1):
        wi = w_dp[i-1]
        vi = v_dp[i-1]
        # Optimization: only update up to cap_int
        # dp[i, :] = dp[i-1, :]
        # if wi <= cap_int:
        #    dp[i, wi:] = np.maximum(dp[i, wi:], dp[i-1, :cap_int-wi+1] + vi)
        
        # Simple loop for clarity and safety, 125k is small.
        # But let's try to be slightly faster with numpy if possible.
        # Actually, a pure python loop for 125k iterations is ~0.01s.
        # Let's stick to the loop for correctness.
        
        # Copy previous row
        dp_table[i, :] = dp_table[i-1, :]
        
        if wi <= cap_int:
            # For j from wi to cap_int:
            # dp_table[i, j] = max(dp_table[i, j], dp_table[i-1, j-wi] + vi)
            
            # Vectorized update:
            # candidates = dp_table[i-1, :cap_int-wi+1] + vi
            # current = dp_table[i, wi:]
            # dp_table[i, wi:] = np.maximum(current, candidates)
            
            candidates = dp_table[i-1, :cap_int-wi+1] + vi
            current = dp_table[i, wi:]
            dp_table[i, wi:] = np.maximum(current, candidates)
            
        # Time check
        if i % 20 == 0 and time.time() > deadline:
            break
    
    # Check if we finished DP
    if time.time() > deadline:
        return _fractional_heuristic(remaining_capacity, weights, values, feasible_indices, start_time, deadline)
    
    # Backtrack to find the first item taken
    # Start from i = n_dp, j = cap_int
    # We want the index in the *original* arrays.
    # The DP was run on `feasible_indices`.
    # So if DP says item k (in the DP array) is taken, the original index is feasible_indices[k].
    
    i = n_dp
    j = cap_int
    selected_dp_indices = []
    
    while i > 0:
        # Check if item i-1 was taken
        # dp_table[i, j] != dp_table[i-1, j] means item was taken
        # But we need to be careful with the vectorized update logic.
        # The condition `dp_table[i, j] == dp_table[i-1, j-wi] + vi` and `dp_table[i-1, j-wi] + vi > dp_table[i-1, j]` is the strict condition.
        # If they are equal, either could be taken. We prefer the one that was taken in the forward pass?
        # The forward pass kept the max. If they are equal, it kept the "not take" value if it was initialized that way?
        # `np.maximum(current, candidates)`: if equal, it keeps `current` (which is `dp_table[i, j]` initially copied from `dp_table[i-1, j]`).
        # So if values are equal, it implies "not take" was preferred (or it was a tie and we kept the old value).
        # So `dp_table[i, j] != dp_table[i-1, j]` is a safe check for "strictly better by taking".
        # If they are equal, we might have a tie. We can arbitrarily pick one.
        
        wi = w_dp[i-1]
        
        if j >= wi and dp_table[i, j] == dp_table[i-1, j-wi] + v_dp[i-1]:
            # It's possible this item was taken.
            # But if dp_table[i, j] == dp_table[i-1, j] as well, it's a tie.
            # Let's check if taking it leads to a valid state.
            selected_dp_indices.append(i-1)
            j -= wi
            i -= 1
        else:
            i -= 1
    
    if not selected_dp_indices:
        return -1
    
    # We found a set of items.
    # Which one to return?
    # The problem asks for the "next" item.
    # In a 0/1 knapsack built incrementally, the order doesn't affect the final set, only the capacity remaining.
    # However, picking any item from the optimal set is a valid move towards the optimal solution.
    # Let's return the first one we found in the backtrack (which corresponds to the last item added in the DP forward pass? No, backtrack goes from end to start).
    # The backtrack finds items in reverse order of addition.
    # It doesn't matter which one we pick, as long as it's in the optimal set.
    # Let's pick the one with the highest value/weight ratio from the selected set?
    # Or just the first one.
    
    # Let's pick the first one in the selected list.
    dp_idx = selected_dp_indices[0]
    original_idx = feasible_indices[dp_idx]
    
    return int(original_idx)


def _fractional_heuristic(remaining_capacity, weights, values, feasible_indices, start_time, deadline):
    """Fallback heuristic: Fractional Knapsack Lookahead."""
    
    n = len(weights)
    
    # Compute densities for all items
    densities = np.zeros(n)
    for i in range(n):
        if weights[i] > 0:
            densities[i] = values[i] / weights[i]
        else:
            densities[i] = float('inf') if values[i] > 0 else 0.0
    
    # Sort indices by density descending
    sorted_indices = np.argsort(-densities)
    
    best_score = -1.0
    best_idx = -1
    
    for i in feasible_indices:
        w_i = weights[i]
        v_i = values[i]
        new_cap = remaining_capacity - w_i
        
        if new_cap < -1e-12:
            continue
        
        # Compute fractional knapsack value on all items except i, with capacity new_cap
        frac_val = 0.0
        current_cap = new_cap
        for idx in sorted_indices:
            if idx == i:
                continue
            if current_cap <= 1e-12:
                break
            w_idx = weights[idx]
            v_idx = values[idx]
            if w_idx <= current_cap + 1e-12:
                frac_val += v_idx
                current_cap -= w_idx
            else:
                frac = current_cap / w_idx if w_idx > 0 else 0.0
                frac_val += frac * v_idx
                current_cap = 0.0
                break
        
        total_score = v_i + frac_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
        
        # Time check
        if time.time() > deadline:
            break
    
    if best_idx < 0:
        # Fallback to best density
        best_density = -1.0
        for i in feasible_indices:
            if densities[i] > best_density:
                best_density = densities[i]
                best_idx = i
    
    return int(best_idx) if best_idx >= 0 else -1
