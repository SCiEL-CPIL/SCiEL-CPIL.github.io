import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety
    
    # 1. Greedy Solution to get an initial upper bound
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
    
    # Lower bound: We need total_diffs distinct values in [1, M]
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    L = total_diffs
    U = best_scope - 1
    
    # Precompute binomial coefficients for pruning
    # binom(n, 2) for n up to k+1
    def binom2(n):
        return n * (n - 1) // 2

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
            
            # Pruning: Check if enough room in [1, M] for remaining differences
            # Diffs used so far
            used_count = len(used_diffs)
            
            # Diffs needed from current row (remaining elements)
            curr_len = len(current_row)
            curr_needed = binom2(k + 1) - binom2(curr_len)
            
            # Diffs needed from future rows
            future_rows = n_rows - row_idx - 1
            future_needed = future_rows * diffs_per_row
            
            total_needed_from_now = curr_needed + future_needed
            
            if used_count + total_needed_from_now > M:
                return False, None
            
            # Optimization: Try smaller x first? Or larger? 
            # Usually, keeping numbers small helps keep differences small, 
            # but we need distinct differences. 
            # Greedy found a solution with scope best_scope. 
            # We are searching for M < best_scope.
            
            for x in range(last + 1, M + 1):
                # Quick check: x - first element must be <= M
                # Since x <= M and first element >= 0, x - first element <= M is always true if x <= M.
                # But we need x - a <= M for all a in current_row.
                # Max diff is x - current_row[0].
                if x - current_row[0] > M:
                    break
                
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
