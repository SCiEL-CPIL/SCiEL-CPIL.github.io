import time
import random
import sys

def construct(n_rows, k):
    # Increase recursion limit for safety
    sys.setrecursionlimit(10000)

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
    
    # Get a greedy upper bound to know how high we might need to go
    greedy_rows = greedy_construct()
    greedy_max = max(max(row) for row in greedy_rows)
    
    best_solution = greedy_rows
    best_max = greedy_max

    # Search for better solutions starting from LB
    # We will search up to greedy_max
    for M in range(lb, greedy_max + 1):
        if time.time() - start_time > time_limit - 5:
            break
            
        # If M is too small to fit even one row, skip (though lb should handle this)
        
        used = set()
        rows = []
        found = False
        attempts = 0
        max_attempts = 100 # Random restarts for this M
        
        def dfs(row_idx, curr_row):
            nonlocal found
            if found:
                return True
            
            if time.time() - start_time > time_limit:
                return False
            
            if len(curr_row) == k + 1:
                if row_idx == n_rows - 1:
                    rows.append(list(curr_row))
                    found = True
                    return True
                return dfs(row_idx + 1, [0])
            
            last = curr_row[-1]
            start_val = last + 1
            
            # Candidates: x in [start_val, M]
            candidates = list(range(start_val, M + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                if found:
                    return True
                new_diffs = {x - a for a in curr_row}
                
                if new_diffs & used:
                    continue
                
                used.update(new_diffs)
                curr_row.append(x)
                
                if dfs(row_idx, curr_row):
                    return True
                
                curr_row.pop()
                used.difference_update(new_diffs)
                
            return False

        # Try a few times with different random seeds for this M
        for _ in range(max_attempts):
            if found:
                break
            if time.time() - start_time > time_limit - 5:
                break
            # Reset state
            used = set()
            rows = []
            found = False
            dfs(0, [0])
            
        if found:
            best_solution = rows
            best_max = M
            # Since we iterate M upwards, the first M with a solution is optimal.
            return best_solution

    return best_solution
