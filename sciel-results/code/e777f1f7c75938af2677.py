import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for overhead
    
    # 1. Greedy Solution
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
        
    best_rows = rows
    best_scope = max(max(r) for r in rows)
    
    # 2. Lower Bound
    total_diffs = n_rows * (k * (k + 1) // 2)
    if best_scope <= total_diffs:
        return best_rows
        
    L = total_diffs
    U = best_scope - 1
    
    # State for backtracking
    timed_out = False
    
    def exists_solution(M):
        nonlocal timed_out
        if timed_out:
            return False, None
            
        used_diffs = set()
        current_rows = []
        
        def backtrack(row_idx, current_row):
            nonlocal timed_out
            if time.time() > deadline:
                timed_out = True
                return False, None
                
            if row_idx == n_rows:
                return True, [r[:] for r in current_rows]
            
            if len(current_row) == k + 1:
                current_rows.append(current_row[:])
                res, sol = backtrack(row_idx + 1, [0])
                if res:
                    return True, sol
                current_rows.pop()
                return False, None
            
            last = current_row[-1]
            # Optimization: Only try x that don't exceed M
            # And ensure diffs are <= M
            for x in range(last + 1, M + 1):
                # Quick check: x - first element must be <= M
                if x - current_row[0] > M:
                    break # Since x is increasing, further x will also violate
                
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    current_row.append(x)
                    used_diffs.update(new_diffs)
                    
                    res, sol = backtrack(row_idx, current_row)
                    if res:
                        return True, sol
                    
                    used_diffs.difference_update(new_diffs)
                    current_row.pop()
            
            return False, None
            
        return backtrack(0, [0])

    while L <= U and not timed_out:
        M = (L + U) // 2
        res, sol = exists_solution(M)
        if res:
            best_rows = sol
            best_scope = M
            U = M - 1
        else:
            if timed_out:
                break
            L = M + 1
            
    return best_rows
