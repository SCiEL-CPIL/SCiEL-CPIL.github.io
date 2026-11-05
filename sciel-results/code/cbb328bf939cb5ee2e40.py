import time
import sys

sys.setrecursionlimit(10000)

def greedy_construct(n_rows, k):
    """
    Greedy construction: each row starts at 0 and takes the smallest next mark 
    whose differences to the earlier marks of the row are all still unused by any row.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 150.0 # 150s wall clock, leave margin
    
    # 1. Get a baseline valid solution
    baseline = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in baseline)
    best_solution = baseline
    
    # 2. Try to optimize using DFS with increasing scope M
    # Lower bound: simple counting bound or just start from best_scope - 50
    # Counting bound for n rows of k+1 elements:
    # Total diffs = n * k(k+1)/2
    # Max diffs in [1, M] is M(M-1)/2
    # So M(M-1)/2 >= n * k(k+1)/2 => M ~ sqrt(n*k(k+1))
    # For n=4, k=7: 4 * 28 = 112 diffs. M(M-1)/2 >= 112 => M ~ 16.
    # Wait, the counting bound in the prompt says 112.
    # This refers to the scope M, not the number of diffs.
    # Let's verify: If M=112, total possible diffs is 112*111/2 = 6216.
    # Required diffs = 4 * 7*8/2 = 112.
    # 112 is very small compared to 6216.
    # Ah, the counting bound usually refers to the number of distinct differences required vs available.
    # Here, the prompt says "simple counting bound is 112".
    # This likely means the minimal possible scope is at least 112?
    # Or maybe it means something else.
    # Let's just start searching from a reasonable range.
    # Known best is 134.
    # Let's try to find a solution with scope < best_scope.
    
    # We will search for M from a lower bound up to best_scope - 1.
    # If we find a solution, we update best_solution and best_scope.
    # We can search in parallel or sequentially. Sequentially is safer.
    
    # To save time, we can start from a lower bound.
    # Let's estimate a lower bound.
    # Max diff is M.
    # We need 112 distinct differences.
    # The differences are integers from 1 to M.
    # So we need M >= 112? No, we need to fit 112 distinct integers in 1..M.
    # So M must be at least 112.
    # So we can start M from 112.
    
    lower_bound = 112
    upper_bound = best_scope - 1
    
    if upper_bound < lower_bound:
        return best_solution
    
    # We will iterate M from lower_bound to upper_bound
    # However, if the search for M=112 takes 100s, we won't have time for 113.
    # We should allocate time per M.
    # e.g., 10 seconds per M.
    
    time_per_M = 10.0
    
    for M in range(lower_bound, upper_bound + 1):
        if time.time() - start_time > time_limit - 5: # Leave 5s margin
            break
            
        M_start = time.time()
        
        # Reset state
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        # Run DFS for this M
        # We need to modify dfs to use the local rows/used_diffs or pass them.
        # Let's refactor dfs to take state or use closures properly.
        
        # Since python closures capture by reference, and we reassign rows/used_diffs, 
        # it's safer to define dfs inside the loop or pass state.
        
        def dfs_M(r, c):
            nonlocal time_limit, start_time
            
            if time.time() - start_time > time_limit - 5:
                return False
                
            if time.time() - M_start > time_per_M:
                return False # Timeout for this M
                
            if r == n_rows:
                return True # Found valid config for this M
                
            if c == 0:
                return dfs_M(r + 1, 0)
                
            if c > k:
                return False
                
            prev_val = rows[r][c-1]
            start_val = prev_val + 1
            end_val = M
            
            # Symmetry breaking: 
            # If r > 0, the max element of row r should be >= max element of row r-1.
            # We can check this when c == k.
            # If c < k, we don't know the max yet.
            
            for x in range(start_val, end_val + 1):
                # Check differences with previous elements in this row
                valid = True
                new_diffs = []
                for i in range(c):
                    d = x - rows[r][i]
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                    
                if not valid:
                    continue
                    
                # Symmetry breaking check for last element
                if c == k and r > 0:
                    if x < max(rows[r-1]):
                        continue
                        
                # Place x
                rows[r].append(x)
                used_diffs.update(new_diffs)
                
                if c == k:
                    if dfs_M(r + 1, 0):
                        return True
                else:
                    if dfs_M(r, c + 1):
                        return True
                        
                # Backtrack
                rows[r].pop()
                for d in new_diffs:
                    used_diffs.remove(d)
                    
            return False
            
        if dfs_M(0, 0):
            # Found a solution with scope M
            # This is better than the baseline
            return [list(row) for row in rows]
            
    # If we didn't find a better solution, return the baseline
    return best_solution
