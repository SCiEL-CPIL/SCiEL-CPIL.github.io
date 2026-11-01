import random
import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    row_len = k + 1
    diffs_per_row = row_len * (row_len - 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    lower_bound = total_diffs_needed
    
    start_time = time.time()
    timeout = 115
    
    # Try to find the smallest M
    # Search range: lower_bound to lower_bound + 50
    # If not found, expand or use fallback
    
    for M in range(lower_bound, lower_bound + 60):
        if time.time() - start_time > timeout:
            break
            
        result = solve_backtrack(n_rows, k, M, start_time, timeout)
        if result is not None:
            return result
            
    # Fallback
    return greedy_fallback(n_rows, k)

def solve_backtrack(n_rows, k, M, start_time, timeout):
    row_len = k + 1
    diffs_per_row = row_len * (row_len - 1) // 2
    
    import random
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        rows = [[0] for _ in range(n_rows)]
        used_mask = 0
        
        def backtrack(row_idx, point_idx):
            if time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                return True
                
            row = rows[row_idx]
            
            # Pruning: Remaining diffs capacity
            used_count = used_mask.bit_count()
            
            rem_diffs_current = 0
            # We are at point_idx (0-indexed index in the final list we are filling)
            # The current row has length point_idx + 1? 
            # No, rows[row_idx] is updated as we go.
            # Initially rows[r] = [0]. point_idx = 1.
            # We are choosing the element at index 1.
            # Elements to be chosen: indices point_idx+1 ... row_len-1.
            # Element at index j adds j diffs.
            for j in range(point_idx + 1, row_len):
                rem_diffs_current += j
                
            rem_diffs_future = (n_rows - row_idx - 1) * diffs_per_row
            
            if used_count + rem_diffs_current + rem_diffs_future > M:
                return False
            
            # Symmetry breaking
            min_val = row[-1] + 1
            if point_idx == 1 and row_idx > 0:
                min_val = max(min_val, rows[row_idx - 1][1] + 1)
                
            max_val = M
            
            if min_val > max_val:
                return False
                
            candidates = []
            for x in range(min_val, max_val + 1):
                new_diffs_mask = 0
                valid = True
                for y in row:
                    d = x - y
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_diffs_mask |= bit
                    
                if valid:
                    candidates.append((x, new_diffs_mask))
                    
            if not candidates:
                return False
                
            # Heuristic ordering
            if restart % 2 == 0:
                candidates.sort(key=lambda t: t[0])
            else:
                random.shuffle(candidates)
                
            for x, new_diffs_mask in candidates:
                row.append(x)
                old_mask = used_mask
                used_mask = old_mask | new_diffs_mask
                
                if backtrack(row_idx, point_idx + 1):
                    return True
                    
                row.pop()
                used_mask = old_mask
                
            return False
            
        if backtrack(0, 1):
            return rows
            
    return None

def greedy_fallback(n_rows, k):
    row_len = k + 1
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < row_len:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used.update(new_diffs)
                    break
                x += 1
        rows.append(row)
    return rows
