import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0  # 10s margin
    
    total_diffs = n_rows * (k * (k + 1) // 2)
    min_m = total_diffs
    
    # Initial greedy solution
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
    best_rows = [r[:] for r in rows]
    best_m = max(max(r) for r in best_rows)
    
    # If greedy is already optimal (m == min_m), return
    if best_m == min_m:
        return best_rows
        
    # Try to find better solutions
    # We iterate m from min_m to best_m - 1
    for m in range(min_m, best_m):
        if time.time() - start_time > time_limit:
            break
            
        # Allocate some time for this m
        # Simple: just run and check time inside
        solution = solve_csp(n_rows, k, m, time_limit - (time.time() - start_time))
        if solution is not None:
            return solution
            
    return best_rows

def solve_csp(n_rows, k, m, time_limit):
    if time_limit <= 0:
        return None
        
    start_time = time.time()
    used_diffs = 0
    rows = [[0] for _ in range(n_rows)]
    
    # Heuristic: Fill rows in order.
    # Symmetry: x[i][1] < x[i+1][1]
    
    def backtrack(row_idx, col_idx):
        nonlocal used_diffs
        
        if time.time() - start_time > time_limit:
            return False
            
        if row_idx == n_rows:
            return True
            
        if col_idx == k + 1:
            return backtrack(row_idx + 1, 1)
            
        row = rows[row_idx]
        
        if col_idx == 1:
            if row_idx == 0:
                start_val = 1
            else:
                start_val = rows[row_idx - 1][1] + 1
            end_val = m
        else:
            start_val = row[col_idx - 1] + 1
            end_val = m
            
        # Optional: Skip values that are too large to fit remaining diffs?
        # Not easy to compute.
        
        for v in range(start_val, end_val + 1):
            valid = True
            new_diffs = 0
            # Check conflicts
            for i in range(col_idx):
                d = v - row[i]
                bit = 1 << d
                if used_diffs & bit:
                    valid = False
                    break
                new_diffs |= bit
            
            if not valid:
                continue
                
            if used_diffs & new_diffs:
                continue
                
            # Commit
            row.append(v)
            old_used = used_diffs
            used_diffs |= new_diffs
            
            if backtrack(row_idx, col_idx + 1):
                return True
                
            # Uncommit
            row.pop()
            used_diffs = old_used
            
        return False
        
    # Special case k=0
    if k == 0:
        return [[0] for _ in range(n_rows)]
        
    result = backtrack(0, 1)
    if result:
        return [r[:] for r in rows]
    return None
