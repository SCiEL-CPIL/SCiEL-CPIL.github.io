import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if len(new_diffs) < len(row):
                    continue
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)
    
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    min_scope = total_diffs
    
    def search(M):
        used_diffs = set()
        rows = []
        prev_row = None
        
        def dfs_row_index(row_idx):
            if row_idx == n_rows:
                return rows[:]
            
            if time.time() > deadline:
                return None
                
            current_row = [0]
            return dfs_build_row(1, current_row, prev_row)

        def dfs_build_row(pos, current_row, prev_row):
            if pos == k + 1:
                # Row complete
                diffs = set()
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        diffs.add(current_row[j] - current_row[i])
                
                if diffs & used_diffs:
                    return None
                
                # Symmetry breaking: ensure this row is lexicographically >= previous row
                if prev_row is not None:
                    if tuple(current_row) < tuple(prev_row):
                        return None
                
                new_used = used_diffs | diffs
                new_rows = rows + [current_row]
                
                # Save state for backtracking? 
                # We are modifying `used_diffs` and `rows` in the closure.
                # It's safer to pass them or use nonlocal with careful restoration.
                # For simplicity in this snippet, let's pass them as args to avoid side effects issues in recursion.
                
                # Re-structure to pass state
                return dfs_row_index_next(row_idx + 1, new_used, new_rows, current_row)

            # Try next mark
            last_mark = current_row[-1]
            for x in range(last_mark + 1, M + 1):
                new_diffs_partial = {x - a for a in current_row}
                if len(new_diffs_partial) < len(current_row):
                    continue
                if new_diffs_partial & used_diffs:
                    continue
                
                # Check symmetry? 
                # We can't fully check symmetry until the row is complete.
                
                # Recurse
                result = dfs_build_row_next(pos + 1, current_row + [x], prev_row)
                if result is not None:
                    return result
            return None
            
        # The nested functions above need to share state or pass it.
        # Let's rewrite search(M) cleanly with explicit state passing.
        
        def dfs_state(row_idx, used_diffs, rows, prev_row):
            if row_idx == n_rows:
                return rows
            
            if time.time() > deadline:
                return None
                
            current_row = [0]
            return dfs_build_state(1, current_row, used_diffs, rows, prev_row)

        def dfs_build_state(pos, current_row, used_diffs, rows, prev_row):
            if pos == k + 1:
                diffs = set()
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        diffs.add(current_row[j] - current_row[i])
                
                if diffs & used_diffs:
                    return None
                
                if prev_row is not None and tuple(current_row) < tuple(prev_row):
                    return None
                
                new_used = used_diffs | diffs
                new_rows = rows + [current_row]
                
                return dfs_state(row_idx + 1, new_used, new_rows, current_row)
            
            last_mark = current_row[-1]
            for x in range(last_mark + 1, M + 1):
                new_diffs_partial = {x - a for a in current_row}
                if len(new_diffs_partial) < len(current_row):
                    continue
                if new_diffs_partial & used_diffs:
                    continue
                
                result = dfs_build_state(pos + 1, current_row + [x], used_diffs, rows, prev_row)
                if result is not None:
                    return result
            return None

        return dfs_state(0, set(), [], None)

    for M in range(min_scope, best_scope):
        if time.time() > deadline:
            break
        solution = search(M)
        if solution is not None:
            return solution
            
    return best_rows
