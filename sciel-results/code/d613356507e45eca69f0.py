import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
        n = len(remaining_weights)
        if n == 0 or remaining_capacity <= 0:
            return -1

        weights = np.array(remaining_weights, dtype=float)
        values = np.array(remaining_values, dtype=float)

        # Filter feasible
        feasible_mask = (weights > 0) & (weights <= remaining_capacity)
        if not np.any(feasible_mask):
            # Check for zero weight positive value
            zero_w_pos_v = (weights == 0) & (values > 0)
            if np.any(zero_w_pos_v):
                return int(np.where(zero_w_pos_v)[0][0])
            return -1

        feasible_indices = np.where(feasible_mask)[0]
        f_weights = weights[feasible_mask]
        f_values = values[feasible_mask]
        f_n = len(f_weights)

        # Scale
        scale = 100
        C_int = int(remaining_capacity * scale)
        w_ints = (f_weights * scale).astype(int)
        
        # Handle rounding issues: if w_int is 0 but weight > 0, set to 1
        w_ints[w_ints == 0] = 1
        
        # DP
        # dp[i][c] = max value using items 0..i-1 with capacity c
        # Size: (f_n + 1) x (C_int + 1)
        dp = np.zeros((f_n + 1, C_int + 1), dtype=float)
        take = np.zeros((f_n + 1, C_int + 1), dtype=bool)

        for i in range(1, f_n + 1):
            w = w_ints[i-1]
            v = f_values[i-1]
            # Copy previous row
            dp[i] = dp[i-1]
            take[i] = take[i-1] # Not strictly needed for take, we can recompute or store
            
            # Update for capacity c from w to C_int
            # dp[i][c] = max(dp[i-1][c], dp[i-1][c-w] + v)
            
            # Vectorized update
            # For c >= w:
            # val_take = dp[i-1][c-w] + v
            # if val_take > dp[i-1][c], then take[i][c] = True, dp[i][c] = val_take
            
            if w <= C_int:
                # c ranges from w to C_int
                # dp[i-1][c-w] corresponds to indices 0 to C_int-w
                # We add v to that slice
                take_vals = dp[i-1][C_int-w::-1] + v # Reversed? No.
                # dp[i-1] is array of length C_int+1.
                # We want to compare dp[i-1][w:] with dp[i-1][:-w] + v
                
                prev_row = dp[i-1]
                take_candidates = prev_row[:C_int-w+1] + v
                current_vals = prev_row[w:]
                
                # If take_candidates > current_vals, we take
                # Note: if equal, we can choose either. Choosing 'take' might lead to smaller indices in traceback?
                # Let's prefer 'take' if equal to find items earlier in the list? Or doesn't matter.
                mask_take = take_candidates > current_vals
                # Also handle equality? If equal, we can set take to False to prefer items that are NOT taken in this branch?
                # Actually, for traceback, we just need *one* path.
                
                dp[i][w:] = np.where(mask_take, take_candidates, current_vals)
                take[i][w:] = mask_take
                # For c < w, take is False (already 0 from init or copy)
                # Wait, take[i] was copied from take[i-1]. We need to reset it or only set where we updated.
                # The copy `take[i] = take[i-1]` is wrong because take[i][c] depends on i.
                # Let's just initialize take[i] to False and set True where we take.
                
        # Re-implement loop for clarity and correctness of `take`
        dp = np.zeros((f_n + 1, C_int + 1), dtype=float)
        take = np.zeros((f_n + 1, C_int + 1), dtype=bool)
        
        for i in range(1, f_n + 1):
            w = w_ints[i-1]
            v = f_values[i-1]
            dp[i] = dp[i-1] # Start with not taking i
            
            if w <= C_int:
                # c from w to C_int
                # Compare dp[i-1][c] vs dp[i-1][c-w] + v
                # dp[i-1][c-w] is dp[i-1][0 : C_int-w+1]
                # dp[i-1][c] is dp[i-1][w : C_int+1]
                
                take_vals = dp[i-1][:C_int-w+1] + v
                current_vals = dp[i-1][w:]
                
                # We want to update dp[i][w:] and take[i][w:]
                # If take_vals > current_vals, take
                mask = take_vals > current_vals
                # If equal, we can leave as is (don't take) or take. 
                # Leaving as is means we prefer the solution that didn't take item i.
                # This is fine.
                
                dp[i][w:] = np.where(mask, take_vals, current_vals)
                take[i][w:] = mask
                # Note: take[i][0:w] remains False (default)
        
        # Traceback
        # Start at i = f_n, c = C_int
        c = C_int
        for i in range(f_n, 0, -1):
            if take[i][c]:
                # Item i-1 is taken
                return int(feasible_indices[i-1])
            else:
                # Item i-1 is not taken
                # c remains same
                pass
                
        # If no item was taken? (Value 0 or all items infeasible in DP due to scaling?)
        # Fallback to greedy
        densities = f_values / f_weights
        best_idx = np.argmax(densities)
        return int(feasible_indices[best_idx])
