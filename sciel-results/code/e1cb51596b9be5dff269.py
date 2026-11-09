import sys
import time

def construct(n_rows, k):
    # Best known for (4, 7) is 134. 
    # We will search for a solution with max < 134.
    # If found, great. If not, we return the best found.
    
    # Initial greedy solution to set upper bound
    used = 0
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            mask = 0
            conflict = False
            for a in row:
                d = x - a
                bit = 1 << d
                if used & bit:
                    conflict = True
                    break
                mask |= bit
            if conflict:
                continue
            row.append(x)
            used |= mask
        greedy_rows.append(row)
    
    best_max = max(max(r) for r in greedy_rows)
    best_solution = [r[:] for r in greedy_rows]
    
    # DFS with Bitsets and Symmetry Breaking
    # Symmetry: row[i][1] >= row[i-1][1]
    # This assumes we construct rows in a specific order.
    
    # To make the search faster, we can limit the time.
    start_time = time.time()
    timeout = 110.0 # 120s limit, leave 10s buffer
    
    def dfs(current_rows, used_mask, current_row, idx, prev_first_elem):
        nonlocal best_solution, best_max, start_time, timeout
        
        if time.time() - start_time > timeout:
            return False # Indicate timeout
        
        if len(current_rows) == n_rows:
            current_max = max(max(row) for row in current_rows)
            if current_max < best_max:
                best_max = current_max
                best_solution = [row[:] for row in current_rows]
            return True # Continue searching for better? Or stop?
            # Since we want the minimum, we should continue if we find one,
            # but we prune based on best_max.
            
        if idx < k:
            # Building current row
            if current_row:
                last = current_row[-1]
                # Min final value for this row
                # We need k - idx more elements.
                # Min increment is 1.
                min_final = last + (k - idx)
                if min_final >= best_max:
                    return True # Pruned, no solution here better than best_max
            
            start_val = current_row[-1] + 1 if current_row else 1
            
            # Symmetry breaking for the first element of the row (idx == 0)
            # If this is the first element (index 1 in 0-indexed list, so idx 0 in loop),
            # it must be >= prev_first_elem.
            if idx == 0:
                if start_val < prev_first_elem:
                    start_val = prev_first_elem
            
            for x in range(start_val, best_max):
                # Check conflict
                mask = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        conflict = True
                        break
                    mask |= bit
                
                if conflict:
                    continue
                
                current_row.append(x)
                
                # Recurse
                # If we just finished the row (idx+1 == k), pass current_row[1] as new prev_first_elem
                next_prev = current_row[1] if (idx + 1 == k) else prev_first_elem
                dfs(current_rows, used_mask | mask, current_row, idx + 1, next_prev)
                
                current_row.pop()
                
        else:
            # Finished a row
            if current_row[-1] >= best_max:
                return True
            
            # Add row
            current_rows.append(current_row[:])
            
            # Start next row
            # The first element of the new row will be constrained by current_row[1]
            dfs(current_rows, used_mask, [0], 0, current_row[1])
            
            current_rows.pop()

    dfs([], 0, [0], 0, 0)
    
    return best_solution
