import time
import random
import math
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    total_diffs = n_rows * (k + 1) * k // 2
    
    # Get an initial upper bound using a fast greedy method
    upper_bound = _greedy_upper_bound(n_rows, k)
    
    best_rows = None
    best_scope = upper_bound
    
    # We search for the minimum M such that a valid configuration exists.
    # We start from a reasonable upper bound and go down.
    # If the greedy bound is very high, we might miss the optimum if we start too high 
    # and don't have time to check all lower M.
    # However, for n=7, k=5, the range is small (105-120).
    
    # Cap the search start to avoid wasting time if greedy is bad
    search_start = min(upper_bound, 120)
    search_end = total_diffs
    
    # Iterate M from search_start down to search_end
    # We allocate a time slice for each M.
    # If we find a solution, we keep it and try lower M.
    
    current_M = search_start
    while current_M >= search_end:
        if time.time() > deadline - 5:
            break
        
        # Time budget for this M
        # If we have plenty of time, give more.
        # Simple heuristic: 2 seconds per M, or remaining time / remaining M's
        remaining_time = deadline - 5 - time.time()
        remaining_ms = search_start - current_M + 1
        time_per_m = max(1.0, remaining_time / remaining_ms)
        
        rows = _dfs_solve(n_rows, k, current_M, start_time + time_per_m)
        
        if rows is not None:
            best_rows = rows
            best_scope = current_M
            # Continue to check if smaller M is possible
        else:
            # If no solution, try next lower M
            pass
        
        current_M -= 1
        
        # If we found a solution, we might want to prioritize finding a better one.
        # The loop naturally does this.
        
    if best_rows is None:
        # Fallback if DFS failed for all M (e.g. timeout on all)
        best_rows = _greedy_construct(n_rows, k)
        best_scope = _scope(best_rows)

    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _greedy_upper_bound(n_rows, k):
    """
    Quick greedy construction to get an upper bound for the search.
    """
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    for j in range(1, k + 1):
        for i in range(n_rows):
            last_val = rows[i][-1]
            x = last_val + 1
            while True:
                new_diffs = {x - a for a in rows[i]}
                if not (new_diffs & used_diffs):
                    break
                x += 1
            rows[i].append(x)
            used_diffs.update(new_diffs)
            
    return max(r[-1] for r in rows)

def _dfs_solve(n_rows, k, max_scope, deadline):
    """
    Depth First Search to find a valid configuration with max element <= max_scope.
    Returns the list of rows if found, else None.
    """
    used_diffs = set()
    rows = []
    
    def is_valid_value(row, val):
        for a in row:
            d = val - a
            if d in used_diffs:
                return False
        return True
        
    def backtrack(row_idx, current_row):
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            return True
            
        if len(current_row) == k + 1:
            rows.append(current_row)
            if backtrack(row_idx + 1, [0]):
                return True
            rows.pop()
            return False
            
        last_val = current_row[-1]
        
        # Pruning: Ensure enough space for remaining elements
        if max_scope - last_val < (k + 1 - len(current_row)):
            return False
            
        for val in range(last_val + 1, max_scope + 1):
            if not is_valid_value(current_row, val):
                continue
                
            current_row.append(val)
            new_diffs = set()
            for a in current_row[:-1]:
                new_diffs.add(val - a)
            used_diffs.update(new_diffs)
            
            if backtrack(row_idx, current_row):
                return True
            
            used_diffs.difference_update(new_diffs)
            current_row.pop()
            
        return False
        
    result = backtrack(0, [0])
    if result:
        return rows
    return None

def _greedy_construct(n_rows, k):
    """
    Fallback greedy constructor.
    """
    rows = []
    used_diffs = set()
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            x = last_x + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    break
                x += 1
            row.append(x)
            used_diffs.update(new_diffs)
        rows.append(row)
    return rows
