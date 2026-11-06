import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
        n = len(remaining_weights)
        if n == 0:
            return -1
        
        weights = np.asarray(remaining_weights, dtype=np.float64)
        values = np.asarray(remaining_values, dtype=np.float64)
        
        # Filter feasible items first to reduce N
        feasible_mask = weights <= remaining_capacity
        if not np.any(feasible_mask):
            return -1
        
        feasible_indices = np.where(feasible_mask)[0]
        f_weights = weights[feasible_indices]
        f_values = values[feasible_indices]
        f_n = len(f_weights)
        
        # Scale
        scale = 1000
        C_int = int(remaining_capacity * scale)
        w_int = (f_weights * scale).astype(int)
        
        # DP
        # dp[j] = max value with capacity j
        dp = np.zeros(C_int + 1, dtype=np.float64)
        # backtrack[i][j] = True if item i is used in optimal solution for capacity j
        # We need to store this for backtracking.
        # Memory: f_n * (C_int + 1) booleans.
        # 100 * 12500 = 1,250,000 booleans. ~1.2 MB. OK.
        backtrack = np.zeros((f_n, C_int + 1), dtype=bool)
        
        for i in range(f_n):
            wi = w_int[i]
            vi = f_values[i]
            if wi == 0:
                # If weight is 0, we can always take it.
                # It will be part of optimal if value > 0.
                # Update dp: dp[j] += vi ? No, 0/1 knapsack.
                # If wi==0, we take it if vi > 0.
                # It doesn't consume capacity.
                # So dp[j] = dp[j] + vi (if we take it).
                # But we can only take it once.
                # Since it doesn't reduce capacity, we should always take it if vi > 0.
                # So we can just add vi to all dp[j] and mark backtrack?
                # Actually, if wi=0, the item is "free".
                # We can just add it to the selected set directly?
                # Let's handle wi=0 separately or assume weights > 0.
                # Given "100-item instances", weights are likely > 0.
                # Let's assume wi > 0. If wi==0, we can just pick it if it's the best?
                # For safety, if wi==0, we can just return this index if vi is high?
                # Let's stick to the loop.
                pass
            
            # Standard 0/1 Knapsack DP
            # Iterate j from C_int down to wi
            # Using numpy slice for speed?
            # dp[wi:] = np.maximum(dp[wi:], dp[:-wi] + vi)
            # But we need backtracking.
            # The vectorized update overwrites dp.
            # We can't easily backtrack from a vectorized update without storing previous states.
            # So we must use the Python loop for backtracking.
            
            # Optimization: Only loop if wi <= C_int
            if wi <= C_int:
                for j in range(C_int, wi - 1, -1):
                    new_val = dp[j - wi] + vi
                    if new_val > dp[j]:
                        dp[j] = new_val
                        backtrack[i][j] = True
                    # If new_val == dp[j], we might have multiple optimal solutions.
                    # We don't strictly need to update backtrack in that case, 
                    # but if we want *an* optimal solution, the current one is fine.
        
        # Backtrack to find selected items
        selected = []
        j = C_int
        for i in range(f_n - 1, -1, -1):
            if backtrack[i][j]:
                selected.append(i)
                j -= w_int[i]
                
        if not selected:
            return -1
        
        # Pick the best item from selected
        # "Best" = highest density?
        # We have indices in 'selected' which are indices into f_weights/f_values.
        # We need to map back to original indices.
        # f_indices[feasible_indices]
        
        best_orig_idx = -1
        best_density = -1.0
        
        for idx_in_f in selected:
            orig_idx = feasible_indices[idx_in_f]
            w = f_weights[idx_in_f]
            v = f_values[idx_in_f]
            if w > 0:
                density = v / w
            else:
                density = float('inf')
            
            if density > best_density:
                best_density = density
                best_orig_idx = orig_idx
                
        return int(best_orig_idx)
