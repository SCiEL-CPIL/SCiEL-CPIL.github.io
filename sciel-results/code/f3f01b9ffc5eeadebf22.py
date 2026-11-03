import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 140.0
    
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lb = total_diffs
    
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows)
    
    best_rows = baseline_rows
    best_scope = baseline_scope
    
    # Only attempt exact search if problem is small
    if total_diffs < 300:
        # Search for solution with scope S
        # We limit the range of S we try to avoid wasting time if LB is far from solution
        # But we want the minimum S.
        # We try S = lb, lb+1, ...
        # If we find one, we stop.
        
        # We can also try to jump to a good S if we have a heuristic?
        # No, just iterate.
        
        for S in range(lb, min(baseline_scope, lb + 100) + 1):
            if time.time() - start_time > time_limit:
                break
            
            solution = backtrack_search_bitmask(n_rows, k, S, start_time, time_limit)
            if solution is not None:
                best_rows = solution
                best_scope = S
                break

    return best_rows

def greedy_construct(n_rows, k):
    used = set()
    rows = []
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

def backtrack_search_bitmask(n_rows, k, max_scope, start_time, time_limit):
    # Bitmask for used differences.
    # Max difference is max_scope.
    # We need bits 1..max_scope.
    # We can use an integer.
    
    used_mask = 0
    rows = []
    
    # Precompute differences for a row? No, they depend on values.
    
    def backtrack(row_idx, current_row):
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            return True
        
        if len(current_row) == k + 1:
            rows.append(current_row)
            return backtrack(row_idx + 1, [0])
        
        last_val = current_row[-1]
        # Next value must be > last_val and <= max_scope
        
        # Optimization: 
        # If we are at the last position (len == k), we can calculate the required value?
        # No, we just iterate.
        
        for x in range(last_val + 1, max_scope + 1):
            # Check differences
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                # d is between 1 and max_scope
                bit = 1 << d
                if used_mask & bit:
                    valid = False
                    break
                new_mask |= bit
            
            if not valid:
                continue
            
            # Pruning: 
            # If we just added x, we used len(current_row) new differences.
            # Total used now: used_mask.bit_count() + len(current_row)
            # This is expensive to compute every time? bit_count is fast in Python 3.10+.
            
            current_row.append(x)
            used_mask_old = used_mask
            used_mask = used_mask_old | new_mask
            
            if backtrack(row_idx, current_row):
                return True
            
            current_row.pop()
            used_mask = used_mask_old
            
        return False

    if backtrack(0, [0]):
        return rows
    return None
