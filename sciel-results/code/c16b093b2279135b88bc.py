import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    # 1. Generate initial solution using greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
        
    current_rows = [row[:] for row in rows]
    best_scope = max(max(row) for row in current_rows)
    best_rows = [row[:] for row in current_rows]
    
    # 2. Local Search
    while time.time() < deadline:
        # Find the row with the largest scope
        max_row_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
        row = current_rows[max_row_idx]
        
        # Compute diffs of other rows
        other_diffs = set()
        for i in range(n_rows):
            if i != max_row_idx:
                other_diffs |= get_diffs(current_rows[i])
        
        improved = False
        
        # Try to reduce elements from the end to the beginning
        for j in range(k, 0, -1):
            old_val = row[j]
            lower_bound = row[j-1] + 1
            upper_bound = row[j+1] - 1 if j < k else old_val - 1
            
            # We want to find the smallest valid new_val in [lower_bound, upper_bound]
            # that is different from old_val.
            # Actually, we want to reduce the scope, so we want new_val < old_val.
            # So we search in [lower_bound, min(upper_bound, old_val - 1)].
            
            search_upper = min(upper_bound, old_val - 1)
            
            found = False
            for new_val in range(lower_bound, search_upper + 1):
                new_row = row[:]
                new_row[j] = new_val
                new_diffs = get_diffs(new_row)
                if not (new_diffs & other_diffs):
                    current_rows[max_row_idx] = new_row
                    improved = True
                    found = True
                    break
            
            if found:
                current_scope = max(max(r) for r in current_rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
                break # Restart to re-evaluate
            
        if not improved:
            # Stuck. Try a random restart?
            # Or just break.
            break
            
    return best_rows
