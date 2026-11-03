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
    
    # Try increasing M
    for M in range(lb, lb + 200): # Search up to 200 above LB
        if time.time() - start_time > time_limit - 10:
            break
            
        # Backtracking search for this M
        used = set()
        rows = []
        
        # Heuristic: Randomize row order? No, rows are identical in structure.
        # But we can randomize the start of the search?
        # Let's just try deterministic small-x first.
        
        def dfs(row_idx, curr_row):
            if time.time() - start_time > time_limit:
                return False
            
            if len(curr_row) == k + 1:
                if row_idx == n_rows - 1:
                    rows.append(list(curr_row))
                    return True
                return dfs(row_idx + 1, [0])
            
            last = curr_row[-1]
            start_val = last + 1
            if not curr_row:
                start_val = 1
            
            # Optimization: We need to pick (k+1 - len(curr_row)) elements.
            # The smallest possible next element is start_val.
            # The largest possible element is M.
            
            # Try values from start_val to M
            # To improve speed, we can shuffle the range if we want, 
            # but for minimization, small values are preferred.
            
            for x in range(start_val, M + 1):
                new_diffs = {x - a for a in curr_row}
                
                # Check conflict with global used
                if new_diffs & used:
                    continue
                
                # Check conflict with internal row diffs?
                # Since we add to 'used' immediately, and 'used' starts empty for the row,
                # we just need to ensure new_diffs are disjoint from 'used'.
                # Wait, 'used' contains diffs from PREVIOUS rows.
                # It does NOT contain diffs from CURRENT row yet.
                # So we must check if new_diffs intersects with diffs ALREADY in curr_row?
                # Let's maintain a local set `row_used` or just check against `used` 
                # if we add to `used` immediately.
                
                # If we add to `used` immediately:
                # 1. Check `new_diffs` against `used` (which includes prev rows + current row so far).
                # 2. If disjoint, add to `used`, recurse, then remove.
                
                if new_diffs & used:
                    continue
                
                used.update(new_diffs)
                curr_row.append(x)
                
                if dfs(row_idx, curr_row):
                    return True
                
                curr_row.pop()
                used.difference_update(new_diffs)
                
            return False

        # Run DFS
        # We might need multiple restarts with different random seeds for the "x" selection?
        # Deterministic small-x is good for finding *small* M solutions.
        
        if dfs(0, [0]):
            return rows

    # If no solution found within time/M limits, return greedy result
    return greedy_construct()
