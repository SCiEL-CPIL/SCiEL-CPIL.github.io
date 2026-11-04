import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # D total diffs
    D = n_rows * k * (k + 1) // 2
    lower_bound = D
    
    # Upper bound from greedy
    def pure_greedy():
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
    
    best_rows = pure_greedy()
    best_scope = max(max(r) for r in best_rows)
    
    def solve_for_scope(M, first_row=None, deadline=None):
        # used_diffs is a bitmask
        used_mask = 0
        rows = []
        
        # Precompute bits for diffs 1..M
        # Note: M <= 134, so 1 << 134 is fine.
        
        def backtrack(row_idx, current_row, used_mask):
            if time.time() > deadline - 2.0:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == 0:
                # Start new row
                # Heuristic: If this is the first row, we can iterate through possible first rows.
                # For subsequent rows, we just find *one* valid row.
                
                # Pruning: Can we complete this row?
                # Not implemented for now, rely on search.
                
                # Try next value
                start_val = current_row[-1] + 1 if current_row else 1
                
                # Iterate candidate values
                # To improve speed, we can limit the search for the first few rows.
                
                # Let's just iterate x from start_val to M
                # We can add a random restart or order change here?
                # For now, deterministic.
                
                for x in range(start_val, M + 1):
                    new_diffs = {x - a for a in current_row}
                    # Check if any diff is already used
                    # Convert new_diffs to bitmask
                    new_mask = 0
                    valid = True
                    for d in new_diffs:
                        if d > M:
                            valid = False
                            break
                        bit = 1 << d
                        if used_mask & bit:
                            valid = False
                            break
                        new_mask |= bit
                    
                    if not valid:
                        continue
                    
                    new_row = current_row + [x]
                    new_used_mask = used_mask | new_mask
                    
                    if len(new_row) == k + 1:
                        # Row complete
                        rows.append(new_row)
                        if backtrack(row_idx + 1, [], new_used_mask):
                            return True
                        rows.pop()
                    else:
                        # Continue row
                        if backtrack(row_idx, new_row, new_used_mask):
                            return True
            
            return False
        
        # Initial call
        if first_row is not None:
            rows = [first_row]
            # Compute mask for first_row
            g_used = 0
            for i in range(len(first_row)):
                for j in range(i + 1, len(first_row)):
                    d = first_row[j] - first_row[i]
                    g_used |= (1 << d)
            return backtrack(1, [], g_used), rows
        else:
            return backtrack(0, [], 0), rows
    
    # Search for better scope
    # We start from lower_bound.
    
    # If n_rows=4, k=7, lower_bound=112.
    # We check M=112, 113, ... up to best_scope.
    
    # To avoid spending too much time on impossible small M,
    # we can add a timeout per M.
    
    # However, backtracking for M=112 might be very fast if it's impossible due to tight constraints,
    # or very slow if it's a "hard" no.
    
    # Let's add a global time check.
    
    # Optimization: If n_rows is small, backtracking is feasible.
    
    # We will try M from lower_bound to best_scope - 1.
    # If we find a solution, we return it.
    
    # Note: The greedy solution is a valid upper bound.
    # We want to minimize.
    
    # To be safe, we'll run backtracking for M in range(lower_bound, best_scope):
    # But we must be careful with time.
    
    # Let's try a few M values.
    
    # If the problem is (4,7), best_scope is likely ~134 from greedy.
    # Range is 112..133. 22 values.
    # If each takes < 1s, we are fine.
    
    for M in range(lower_bound, best_scope):
        if time.time() > deadline - 5.0:
            break
        
        # We can add a small randomization or different heuristic for each M?
        # No, deterministic is better for reproducibility.
        
        success, res_rows = solve_for_scope(M, first_row=None, deadline=deadline)
        if success:
            best_rows = res_rows
            best_scope = M
            # Since we are searching from bottom up, the first one found is the minimum for this search.
            # We can break here if we trust the search.
            # But maybe a smaller M was skipped? No, we iterate sequentially.
            # So this is the minimal M found.
            break
            
    return best_rows
