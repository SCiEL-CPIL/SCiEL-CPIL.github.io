import time
import random
from itertools import combinations

def construct(n_rows, k):
    # Fallback greedy
    def greedy_construct():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    start_time = time.time()
    time_limit = 110.0
    
    # Lower bound
    diffs_per_row = (k + 1) * k // 2
    lb = n_rows * diffs_per_row
    
    # Calculate minimum M based on counting bound
    import math
    M_min = int(math.ceil((math.sqrt(1 + 8 * lb) - 1) / 2))
    
    # Try increasing M
    for M in range(M_min, M_min + 100): # Search up to 100 above counting LB
        if time.time() - start_time > time_limit - 10:
            break
            
        # Backtracking search for this M
        used = set()
        rows = []
        total_diffs_capacity = M * (M + 1) // 2
        
        def dfs(row_idx, curr_row):
            if time.time() - start_time > time_limit:
                return False
            
            # Pruning: Check if remaining capacity is sufficient
            c = len(curr_row)
            diffs_in_curr_row = c * (c - 1) // 2
            diffs_needed_curr = diffs_per_row - diffs_in_curr_row
            rows_after = n_rows - row_idx - 1
            diffs_needed_after = rows_after * diffs_per_row
            diffs_needed_total = diffs_needed_curr + diffs_needed_after
            
            diffs_available = total_diffs_capacity - len(used)
            
            if diffs_available < diffs_needed_total:
                return False
            
            if len(curr_row) == k + 1:
                if row_idx == n_rows - 1:
                    rows.append(list(curr_row))
                    return True
                return dfs(row_idx + 1, [0])
            
            last = curr_row[-1]
            start_val = last + 1
            
            # Optimization: We need to pick (k+1 - len(curr_row)) elements.
            # The smallest possible next element is start_val.
            # The largest possible element is M.
            
            for x in range(start_val, M + 1):
                new_diffs = {x - a for a in curr_row}
                
                # Check conflict with global used
                if new_diffs & used:
                    continue
                
                used.update(new_diffs)
                curr_row.append(x)
                
                if dfs(row_idx, curr_row):
                    return True
                
                curr_row.pop()
                used.difference_update(new_diffs)
                
            return False

        if dfs(0, [0]):
            return rows

    # If no solution found within time/M limits, return greedy result
    return greedy_construct()
