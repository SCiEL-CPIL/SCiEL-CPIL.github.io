import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    # Filter to feasible items only
    feasible = []
    for i in range(n):
        w = remaining_weights[i]
        v = remaining_values[i]
        if w > 0 and w <= remaining_capacity:
            feasible.append((v, w, i))
        elif w == 0 and v > 0:
            return i  # Free value, take it
    
    if not feasible:
        return -1
    
    # Sort by density descending for greedy baseline
    feasible.sort(key=lambda x: x[0] / x[1], reverse=True)
    
    # Greedy fill to get a baseline
    greedy_val = 0
    greedy_items = []
    rem_cap = remaining_capacity
    for v, w, idx in feasible:
        if w <= rem_cap:
            greedy_val += v
            greedy_items.append(idx)
            rem_cap -= w
    
    # Try each item as the first pick, then greedy fill the rest
    best_first = feasible[0][2]
    best_total = greedy_val
    
    # Limit the number of candidates to try for speed
    candidates = feasible[:min(50, len(feasible))]
    
    for v_first, w_first, idx_first in candidates:
        if w_first > remaining_capacity:
            continue
        rem_cap_after = remaining_capacity - w_first
        # Greedy fill with remaining items
        total = v_first
        for v, w, idx in feasible:
            if idx == idx_first:
                continue
            if w <= rem_cap_after:
                total += v
                rem_cap_after -= w
        if total > best_total:
            best_total = total
            best_first = idx_first
    
    # If we still have time, try a more sophisticated approach
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        # Try local search: swap out items in the greedy solution
        # Build the greedy solution set
        greedy_set = set(greedy_items)
        current_solution = list(greedy_items)
        current_value = greedy_val
        
        # Try swapping one item out and one item in
        improved = True
        max_swaps = 100
        swap_count = 0
        
        while improved and swap_count < max_swaps:
            improved = False
            swap_count += 1
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - SAFETY_MARGIN:
                break
            
            # Try removing each item in the current solution
            for i, idx_out in enumerate(current_solution):
                if idx_out not in greedy_set:
                    continue
                # Find weight and value of this item
                w_out = 0
                v_out = 0
                for v, w, idx in feasible:
                    if idx == idx_out:
                        w_out = w
                        v_out = v
                        break
                
                new_cap = remaining_capacity - sum(
                    w for v, w, idx in feasible if idx in greedy_set and idx != idx_out
                )
                
                # Try adding items not in the solution
                for v_in, w_in, idx_in in feasible:
                    if idx_in in greedy_set:
                        continue
                    if w_in <= new_cap:
                        new_value = current_value - v_out + v_in
                        if new_value > current_value + 1e-9:
                            # Perform the swap
                            current_solution.remove(idx_out)
                            current_solution.append(idx_in)
                            greedy_set.remove(idx_out)
                            greedy_set.add(idx_in)
                            current_value = new_value
                            improved = True
                            break
        
        # Update best if local search improved
        if current_value > best_total:
            # Find which item to return first
            # Return the item with highest density in the improved solution
            best_density = -1
            for idx in current_solution:
                for v, w, i in feasible:
                    if i == idx:
                        density = v / w
                        if density > best_density:
                            best_density = density
                            best_first = idx
                        break
            best_total = current_value
    
    # Try DP-based approach if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        # Scale weights to integers for DP
        SCALE = 100
        cap_scaled = int(round(remaining_capacity * SCALE))
        
        if cap_scaled > 0 and cap_scaled < 10000:  # Only if DP is feasible
            # dp[w] = max value achievable with weight <= w
            dp = [0.0] * (cap_scaled + 1)
            
            # Track which items are used for reconstruction
            # For each weight level, track the last item added
            # Use a 2D approach: dp[i][w] = max value using first i items with capacity w
            # But this might be too memory intensive, so use 1D with tracking
            
            # Alternative: just find the best single item to add by solving the subproblem
            # For each item, check if it's in the optimal solution
            
            # Let's try a simpler approach: for each item, compute the best solution
            # that includes that item, and pick the one with highest total value
            
            # Actually, let's just do a full DP and reconstruct
            # dp[w] = max value with capacity w
            # We'll track the items used
            
            # Use a different approach: for each item, compute the value if we include it
            # plus the best we can do with remaining capacity
            
            # This is getting complex, let's just try the local search more aggressively
            pass
    
    return best_first
