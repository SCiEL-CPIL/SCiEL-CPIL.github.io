import sys
import time

def construct(n_rows, k):
    min_M = n_rows * (k + 1) * k // 2
    
    def find_solution(M):
        solution = None
        
        def dfs(rows, used_diffs, current_row, next_idx):
            nonlocal solution
            if solution is not None:
                return True
            
            if len(rows) == n_rows:
                solution = [r[:] for r in rows]
                return True
            
            if next_idx <= k:
                last_val = current_row[-1]
                row_idx = len(rows)
                max_allowed_last = M - (n_rows - 1 - row_idx)
                
                if last_val >= max_allowed_last:
                    return False
                
                # x + (k - next_idx) <= max_allowed_last
                # x <= max_allowed_last - (k - next_idx)
                start_val = last_val + 1
                end_val = max_allowed_last - (k - next_idx) + 1
                
                if start_val >= end_val:
                    return False
                
                for x in range(start_val, end_val):
                    new_diffs = []
                    conflict = False
                    for a in current_row:
                        d = x - a
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.append(d)
                    
                    if conflict:
                        continue
                    
                    current_row.append(x)
                    used_diffs.update(new_diffs)
                    
                    if dfs(rows, used_diffs, current_row, next_idx + 1):
                        return True
                    
                    current_row.pop()
                    used_diffs.difference_update(new_diffs)
                
                return False
            else:
                row_max = current_row[-1]
                if rows and rows[-1][-1] >= row_max:
                    return False
                
                new_rows = rows + [current_row[:]]
                return dfs(new_rows, used_diffs, [0], 1)
        
        dfs([], set(), [0], 1)
        return solution

    start_M = max(min_M, k * (k + 1) // 2)
    
    # Add a safety timeout
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    M = start_M
    while True:
        if time.time() - start_time > timeout:
            break
        sol = find_solution(M)
        if sol is not None:
            return sol
        M += 1
    
    # If timeout, return a greedy solution or the last best effort
    # For now, return a greedy construction to ensure valid output
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
