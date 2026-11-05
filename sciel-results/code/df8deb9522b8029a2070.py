import time
from itertools import combinations

def construct(n_rows, k):
    deadline = time.time() + 145  # 150s limit, 5s margin
    num_marks = k + 1
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    
    best_result = None
    best_max = float('inf')
    
    def backtrack(rows, used_diffs, max_val):
        nonlocal best_result, best_max, deadline
        if time.time() > deadline:
            return False
        
        if len(rows) == n_rows:
            current_max = max(max(row) for row in rows)
            if current_max < best_max:
                best_max = current_max
                best_result = [row[:] for row in rows]
            return True
        return False
    
    # Try different upper bounds starting from lower bound
    # Lower bound: need total_diffs_needed distinct values from 1..M, so M >= total_diffs_needed
    lower_bound = total_diffs_needed
    
    for max_allowed in range(lower_bound, lower_bound + 30):
        if time.time() > deadline:
            break
        
        # Reset for this bound
        used_diffs = set()
        rows = []
        
        def search_row(row_idx, row, pos):
            nonlocal used_diffs, rows, deadline
            if time.time() > deadline:
                return False
            
            if pos == num_marks:
                rows.append(row[:])
                return search_next_row(row_idx + 1)
            
            # Try possible next marks
            # The next mark must be > row[-1]
            start = row[-1] + 1
            # Upper bound: can't exceed max_allowed
            end = min(max_allowed, row[-1] + (num_marks - pos))  # need at least (num_marks-pos) more marks
            
            for candidate in range(start, end + 1):
                # Check if all differences from candidate to existing marks are unused
                new_diffs = {candidate - a for a in row}
                if new_diffs & used_diffs:
                    continue
                
                # Forward checking: can we still place remaining marks?
                # The span from candidate to max_allowed must allow (num_marks - pos - 1) more marks
                if max_allowed - candidate < (num_marks - pos - 1):
                    continue
                
                row.append(candidate)
                used_diffs.update(new_diffs)
                
                if search_row(row_idx, row, pos + 1):
                    return True
                
                row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        def search_next_row(row_idx):
            nonlocal used_diffs, rows, deadline
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            return search_row(row_idx, [0], 0)
        
        if search_next_row(0):
            if best_result is not None:
                return best_result
    
    # Fallback: return greedy if backtracking didn't find anything
    if best_result is not None:
        return best_result
    
    # Greedy fallback
    used = set()
    rows = []
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
