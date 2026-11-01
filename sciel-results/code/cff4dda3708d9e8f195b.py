import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # 1. Greedy solution for fallback and initial UB
    used_greedy, rows_greedy = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used_greedy:
                row.append(x)
                used_greedy |= new
        rows_greedy.append(row)
    
    best_solution = rows_greedy
    best_max = max(max(row) for row in best_solution)
    
    # 2. Search for better solution
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs
    
    def solve(m_limit):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, mark_idx, current_row):
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                return rows[:]
            
            # Determine start for next mark
            if mark_idx == 0:
                current_row = [0]
                # Symmetry: first non-zero of this row > first non-zero of prev row
                if row_idx > 0:
                    min_start = rows[row_idx - 1][1] + 1
                else:
                    min_start = 1
            else:
                min_start = current_row[-1] + 1
            
            # Iterate candidates
            for x in range(min_start, m_limit + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # Additional Pruning: 
                # If we are in the last mark of a row, or last row, check feasibility?
                # Simple check: if x is too large, we might not be able to fill subsequent rows 
                # with smaller numbers if we enforced ordering? No, ordering is only on first marks.
                
                next_row = current_row + [x]
                
                if mark_idx == k:
                    rows.append(next_row)
                    used_diffs.update(new_diffs)
                    
                    result = backtrack(row_idx + 1, 0, [])
                    if result is not None:
                        return result
                    
                    rows.pop()
                    used_diffs.difference_update(new_diffs)
                else:
                    result = backtrack(row_idx, mark_idx + 1, next_row)
                    if result is not None:
                        return result
            
            return None
        
        return backtrack(0, 0, [])

    # Iterate M from LB to best_max - 1
    # We can limit the range if best_max is much larger than LB
    # But for 4,7, LB=112, Best~150. Range is ~40.
    for m in range(lb, best_max):
        if time.time() > deadline:
            break
        
        solution = solve(m)
        if solution is not None:
            # Verify solution validity just in case
            all_diffs = set()
            valid = True
            for row in solution:
                if len(row) != k + 1 or row[0] != 0:
                    valid = False
                    break
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                actual_max = max(max(row) for row in solution)
                if actual_max < best_max:
                    best_solution = solution
                    best_max = actual_max
                    return solution
                else:
                    # If actual_max is not less, it means we found a solution 
                    # but it's not better than greedy? 
                    # This shouldn't happen if m < best_max and solution max <= m.
                    # Unless greedy was very good.
                    # In that case, we continue or return?
                    # If m < best_max, and we found a solution, its max is <= m < best_max.
                    # So it should be better.
                    return solution

    return best_solution
