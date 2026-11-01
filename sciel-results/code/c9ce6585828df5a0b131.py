import time

def construct(n_rows, k):
    start_time = time.time()
    timeout = 140.0  # Leave 10s margin for overhead
    
    # Lower bound from counting argument
    total_diffs = n_rows * k * (k+1) // 2
    M_min = total_diffs
    
    # Fallback greedy solution
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    fallback = greedy_construct()
    
    # Backtracking solver for fixed M
    def solve_for_M(M):
        used = set()
        rows = []
        
        def backtrack(row_idx, cur_row):
            # Check time limit
            if time.time() - start_time > timeout:
                return None
            
            # Pruning: Check if enough differences remain
            # Diffs used so far: len(used)
            # Diffs available in [1, M]: M
            # Diffs needed for remaining rows (including current if not complete? No, current is partial)
            # Let's say we are at row_idx (0-indexed).
            # Rows 0 to row_idx-1 are complete.
            # Row row_idx is cur_row.
            # Remaining rows to complete: n_rows - row_idx
            # But cur_row is not complete yet.
            # Diffs in cur_row so far: len(cur_row)*(len(cur_row)-1)//2
            # Diffs needed to complete cur_row:
            #   Let r = len(cur_row). Next element is r-th (0-indexed).
            #   Diffs added by next element: r
            #   Total diffs in full row: k(k+1)/2
            #   Diffs already in cur_row: r(r-1)/2
            #   Diffs to add to complete cur_row: k(k+1)/2 - r(r-1)/2
            # Diffs needed for rows row_idx+1 to n_rows-1:
            #   (n_rows - 1 - row_idx) * k(k+1)/2
            
            if row_idx == n_rows:
                return rows[:]
            
            if len(cur_row) == k + 1:
                # Row complete
                if row_idx > 0:
                    prev = rows[-1]
                    if cur_row < prev:
                        return None
                
                rows.append(cur_row)
                
                new_diffs = set()
                for i in range(len(cur_row)):
                    for j in range(i+1, len(cur_row)):
                        new_diffs.add(cur_row[j] - cur_row[i])
                
                if new_diffs & used:
                    rows.pop()
                    return None
                
                used.update(new_diffs)
                
                res = backtrack(row_idx + 1, [0])
                if res is not None:
                    return res
                
                used.difference_update(new_diffs)
                rows.pop()
                return None
            
            else:
                last_val = cur_row[-1] if cur_row else 0
                # Bound: need to fit remaining elements
                if last_val + (k + 1 - len(cur_row)) > M:
                    return None
                
                # Pruning: available diffs check
                # Diffs used so far: len(used)
                # Max diffs possible: M
                # Diffs needed:
                #   For current row completion:
                r = len(cur_row)
                diffs_in_cur = r * (r - 1) // 2
                total_diffs_row = k * (k + 1) // 2
                needed_cur = total_diffs_row - diffs_in_cur
                # For future rows:
                future_rows = n_rows - row_idx - 1
                needed_future = future_rows * total_diffs_row
                total_needed = needed_cur + needed_future
                
                if M - len(used) < total_needed:
                    return None
                
                for x in range(last_val + 1, M + 1):
                    conflict = False
                    new_diffs = []
                    for a in cur_row:
                        d = x - a
                        if d in used:
                            conflict = True
                            break
                        new_diffs.append(d)
                    
                    if conflict:
                        continue
                    
                    cur_row.append(x)
                    used.update(new_diffs)
                    
                    res = backtrack(row_idx, cur_row)
                    if res is not None:
                        return res
                    
                    cur_row.pop()
                    used.difference_update(new_diffs)
                
                return None
        
        return backtrack(0, [0])

    # Try M from M_min upwards
    # We cap the search at a reasonable upper bound or until timeout
    # If we find a solution, it is optimal for that M.
    # Since we go up, the first solution is the global optimum.
    
    # To avoid spending all time on infeasible small M, we can limit time per M?
    # Or just rely on the global timeout.
    # The pruning `M - len(used) < total_needed` should make infeasible cases fail fast.
    
    for M in range(M_min, 200):
        if time.time() - start_time > timeout:
            break
        sol = solve_for_M(M)
        if sol is not None:
            return sol
            
    return fallback
