"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
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

    # Attempt to use Dynamic Programming if weights can be scaled to integers
    # Try scaling factors to handle floating point precision and small decimals
    best_dp_index = -1
    dp_success = False
    
    for scale in [10, 100, 1000]:
        weights_scaled = weights * scale
        # Check if weights are effectively integers
        if np.allclose(weights_scaled, np.round(weights_scaled)):
            weights_int = np.round(weights_scaled).astype(int)
            cap_int = int(round(remaining_capacity * scale))
            
            # DP Table: dp[i][c] = max value using first i items with capacity c
            # We only care about feasible items? No, DP considers all, but infeasible ones just won't be picked.
            # Actually, we can filter to feasible items to save space/time, but we need to map back to original indices.
            # Let's just use all items, DP will handle it.
            
            # Optimization: If cap_int is too large, skip DP. 
            # Capacity 12.5 * 1000 = 12500. N=100. 100*12500 = 1.25M states. 
            # Python loops are slow. Vectorized DP or C-optimized? 
            # Standard Python DP might be too slow for 1.25M states if done naively in a loop.
            # But we can use a 1D array for capacity and update backwards.
            
            if cap_int > 20000: # Threshold to avoid timeout in pure Python DP
                continue
                
            # 1D DP array
            dp = [0.0] * (cap_int + 1)
            # To reconstruct the solution, we need to know which items were taken.
            # We can store the previous state or use a 2D boolean table for reconstruction.
            # 2D boolean table: 100 * 12500 = 1.25M booleans. Memory is fine (~1MB).
            # But time? 1.25M operations in Python loop is ~0.1-0.5s. Acceptable.
            
            # Let's try a slightly more efficient approach or just stick to heuristic if DP is risky.
            # Given the 120s limit, and 8 instances, we have ~15s per instance.
            # 1.25M iterations in Python is ~0.1s. So DP is feasible.
            
            # Reconstructing which items are in the optimal set:
            # We need to know for each (i, c) if item i-1 was taken.
            # This requires O(N * C) memory.
            
            # Alternative: Just compute the max value, then backtrack?
            # Backtracking with 1D DP is hard without storing history.
            # Let's use a 2D array of booleans `take` to track decisions.
            
            # However, initializing a 2D array of 100x12500 in Python is slow.
            # Let's try to just use the greedy heuristic if scaling factor > 10.
            # If scale is 10, cap is 125. N=100. 100*125 = 12500 states. Very fast.
            # If scale is 100, cap is 1250. 100*1250 = 125000 states. Fast.
            # If scale is 1000, cap is 12500. 100*12500 = 1.25M states. Might be ~0.5s-1s.
            
            # Let's restrict DP to scale=10 and scale=100 for safety.
            if scale > 100:
                continue
                
            dp = np.zeros(cap_int + 1, dtype=float)
            # For reconstruction, we store the item index that led to the current state?
            # Or we can just run DP to get the max value, and then do a second pass to find which items are essential?
            # Actually, we just need *one* item from the optimal set.
            
            # Simpler reconstruction:
            # Store `dp_prev` and `dp_curr`? No, we need to know *which* items.
            
            # Let's use a dictionary or list of lists for reconstruction if N is small.
            # Since N=100, we can store `choice[i][c]` = 1 if item i-1 is taken.
            choice = np.zeros((n, cap_int + 1), dtype=bool)
            
            for i in range(n):
                w_i = weights_int[i]
                v_i = values[i]
                if w_i > cap_int:
                    # Item i cannot be taken, copy previous
                    # dp[i] = dp[i-1]
                    # choice[i] = False
                    continue
                
                # Update dp from high capacity to low
                for c in range(cap_int, w_i - 1, -1):
                    val_without = dp[c]
                    val_with = dp[c - w_i] + v_i
                    if val_with > val_without:
                        dp[c] = val_with
                        choice[i][c] = True
                    else:
                        choice[i][c] = False
                # For c < w_i, choice[i][c] remains False (default)
            
            # Now backtrack to find which items are in the optimal solution for capacity cap_int
            optimal_value = dp[cap_int]
            # We want to find the first item (lowest index in remaining_weights) that is part of the optimal set.
            # The DP considers items in order 0..n-1.
            # Backtracking starts from (n-1, cap_int).
            # We need to trace back which items were selected.
            
            selected_items = set()
            c = cap_int
            for i in range(n - 1, -1, -1):
                if choice[i][c]:
                    selected_items.add(i)
                    c -= weights_int[i]
                # else: item i was not taken, c stays same
            
            # Find the first item in `remaining_indices` (feasible_indices) that is in `selected_items`
            # Note: `selected_items` contains indices from 0 to n-1.
            # We want the index in the original array.
            # The function returns `index into the remaining (feasible) arrays`?
            # NO: "index into the remaining (feasible) arrays" usually means the index in the input arrays provided to the function.
            # The input arrays are `remaining_weights` and `remaining_values`.
            # So if I return `i`, it refers to `remaining_weights[i]`.
            
            # We need to find the smallest `i` in `feasible_indices` such that `i` is in `selected_items`.
            # Actually, any item in the optimal set is a valid "next item" to pick?
            # If we pick item `i`, we then solve the subproblem for the rest.
            # If the set `S` is optimal, picking any `i` in `S` is a valid move towards the optimal solution.
            # However, the heuristic must be consistent.
            # If we pick `i`, the next call will have `remaining_capacity` reduced by `w_i`.
            # The remaining items will be `remaining_weights` with `i` removed.
            # The DP solution for the subproblem should match the rest of the optimal set.
            
            # So we can just return the smallest index in `selected_items` that is feasible.
            # Since all items in `selected_items` are feasible (weight <= cap), this should be fine.
            
            candidate_indices = [i for i in selected_items]
            if candidate_indices:
                # Return the one with the highest value? Or just the first?
                # Picking the one that leaves the "best" subproblem is ideal.
                # But since the set is optimal, any item in it works.
                # Let's pick the one with the highest value to be safe? 
                # Or the one with the smallest index?
                # The evaluator just checks if the final solution is good.
                # It doesn't care which item we pick, as long as we pick one from the optimal set.
                # So we can just return `min(candidate_indices)`.
                best_dp_index = min(candidate_indices)
                dp_success = True
                break

    if dp_success:
        return int(best_dp_index)

    # Fallback: Greedy by Value/Weight
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, np.inf)
    
    # Only consider feasible items
    feasible_densities = densities[feasible_mask]
    feasible_values = values[feasible_mask]
    
    if len(feasible_densities) == 0:
        return -1
        
    # Find the index in the original array of the best feasible item
    # We need the index in `remaining_weights`
    # `np.argmax` on the masked array gives index in the masked array.
    # We need to map back.
    
    # Create an array of densities for all items, set infeasible to -inf
    all_densities = np.full(n, -np.inf)
    all_densities[feasible_mask] = feasible_densities
    
    best_idx = int(np.argmax(all_densities))
    return best_idx
# EVOLVE-BLOCK-END
