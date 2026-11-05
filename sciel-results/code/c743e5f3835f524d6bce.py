import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs
    
    best_solution = None
    best_max_val = float('inf')
    
    # We will use a list for rows to avoid copying
    rows = []
    current_row = []
    used_diffs = set()
    
    def backtrack(row_idx, current_row, used_diffs):
        nonlocal best_solution, best_max_val, deadline
        
        if time.time() > deadline:
            return
            
        if row_idx == n_rows:
            # Check if this is a better solution
            # We can compute max val
            max_val = 0
            for r in rows:
                if r:
                    m = r[-1] # Since rows are increasing, last element is max
                    if m > max_val:
                        max_val = m
            
            if max_val < best_max_val:
                best_max_val = max_val
                best_solution = [r[:] for r in rows]
            return
        
        if not current_row:
            current_row = [0]
            rows.append(current_row)
        
        if len(current_row) == k + 1:
            # Row complete
            rows.pop() # Remove current row from active list? No, we need to keep it for the final check.
            # Actually, it's better to pass rows as a parameter or manage it carefully.
            # Let's restructure: rows is a list of lists.
            # When we finish a row, we append it to a 'completed_rows' list?
            # Or just keep it in `rows` and pop when backtracking.
            
            # Let's assume `rows` contains completed rows + current_row is the last one?
            # No, let's keep `rows` as the list of *completed* rows.
            # And `current_row` is separate.
            
            # If row is complete:
            # Add to completed rows?
            # Actually, the previous logic had `rows` holding the completed rows.
            # Let's stick to:
            # rows: list of completed rows
            # current_row: list of marks for the row in progress
            
            # When len(current_row) == k+1:
            #   rows.append(current_row)
            #   backtrack(row_idx + 1, [], used_diffs)
            #   rows.pop()
            
            # So in the base case (row_idx == n_rows), `rows` contains all n_rows rows.
            
            return
        
        last_mark = current_row[-1]
        
        # Heuristic: Limit x to best_max_val if we have a solution
        # If no solution, use a heuristic upper bound
        # A good upper bound is lower_bound + k
        max_x_limit = best_max_val if best_solution else (lower_bound + k + 10)
        
        # Optimization: If we have a solution, we can prune x > best_max_val
        # Also, we can prune if x - last_mark is too large? No.
        
        for x in range(last_mark + 1, max_x_limit + 1):
            new_diffs = set()
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            # Recurse
            used_diffs.update(new_diffs)
            current_row.append(x)
            
            # If this is the last row, we can check max_val early?
            # No, just recurse.
            
            backtrack(row_idx, current_row, used_diffs)
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
            # If time is up, return
            if time.time() > deadline:
                return

    # Initialize
    # We need to handle the row completion logic correctly.
    # Let's wrap the logic in a helper or adjust the backtrack signature.
    
    # Revised Backtrack:
    def bt(row_idx, current_row, used_diffs, completed_rows):
        nonlocal best_solution, best_max_val, deadline
        
        if time.time() > deadline:
            return
            
        if row_idx == n_rows:
            # All rows done
            max_val = max(r[-1] for r in completed_rows)
            if max_val < best_max_val:
                best_max_val = max_val
                best_solution = [r[:] for r in completed_rows]
            return
        
        if not current_row:
            current_row = [0]
        
        if len(current_row) == k + 1:
            # Row complete
            completed_rows.append(current_row)
            bt(row_idx + 1, [], used_diffs, completed_rows)
            completed_rows.pop()
            return
        
        last_mark = current_row[-1]
        
        max_x_limit = best_max_val if best_solution else (lower_bound + k + 20)
        
        for x in range(last_mark + 1, max_x_limit + 1):
            new_diffs = set()
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            used_diffs.update(new_diffs)
            current_row.append(x)
            
            bt(row_idx, current_row, used_diffs, completed_rows)
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
            if time.time() > deadline:
                return

    bt(0, [], set(), [])
    
    if best_solution:
        return best_solution
    
    # Fallback Greedy
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
    return rows
