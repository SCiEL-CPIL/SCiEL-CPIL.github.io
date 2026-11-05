import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Calculate lower bound
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # If k is small, we can be more aggressive
    # For the specific case k=7, n=4, LB=112.
    
    def solve_for_scope(M):
        # Backtracking solver
        # rows: list of lists
        # used: int bitmask of used differences
        rows = [[] for _ in range(n_rows)]
        used = 0
        
        # Heuristic: Order rows? 
        # We build row 0, then row 1, etc.
        
        def backtrack(row_idx, col_idx):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            if row_idx == n_rows:
                return True
            
            if col_idx == k + 1:
                # Move to next row
                return backtrack(row_idx + 1, 0)
            
            current_row = rows[row_idx]
            
            # Determine min value
            if col_idx == 0:
                min_val = 0
            else:
                min_val = current_row[-1] + 1
            
            # Determine max value
            # Must leave room for remaining k - col_idx elements
            remaining = k - col_idx
            max_val = M - remaining
            
            if min_val > max_val:
                return False
            
            # Value ordering: Try smaller values first to keep density?
            # Or random?
            # Let's try a mix: iterate, but maybe shuffle if stuck?
            # For now, simple iteration.
            
            # To improve search, we can randomize the order of candidates slightly
            # or restart if stuck.
            # Let's stick to deterministic first, then randomize if needed.
            
            for v in range(min_val, max_val + 1):
                # Check conflicts
                new_diffs = 0
                conflict = False
                
                for u in current_row:
                    d = v - u
                    # Check if d is in used
                    if (used >> d) & 1:
                        conflict = True
                        break
                    # Check if d is in new_diffs (internal conflict)
                    if (new_diffs >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                
                if conflict:
                    continue
                
                # Accept
                current_row.append(v)
                used |= new_diffs
                
                if backtrack(row_idx, col_idx + 1):
                    return True
                
                # Backtrack
                current_row.pop()
                used ^= new_diffs # Restore used (XOR works if we know exactly which bits we set)
                # Wait, used is a local variable in the closure? No, it's in the enclosing scope.
                # We need to be careful with `used`.
                # It's better to pass `used` as an argument or use a global/nonlocal.
                # Let's pass it as an argument to make it pure/functional-ish for backtracking.
                
            return False
        
        # Wrapper to handle `used` passing
        def bt(row_idx, col_idx, used_mask):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            if row_idx == n_rows:
                return True
            
            if col_idx == k + 1:
                return bt(row_idx + 1, 0, used_mask)
            
            current_row = rows[row_idx]
            
            if col_idx == 0:
                min_val = 0
            else:
                min_val = current_row[-1] + 1
            
            remaining = k - col_idx
            max_val = M - remaining
            
            if min_val > max_val:
                return False
            
            # Try values
            # Heuristic: Smallest first
            for v in range(min_val, max_val + 1):
                new_diffs = 0
                conflict = False
                
                for u in current_row:
                    d = v - u
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                    if (new_diffs >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                
                if conflict:
                    continue
                
                current_row.append(v)
                
                if bt(row_idx, col_idx + 1, used_mask | new_diffs):
                    return True
                
                current_row.pop()
            
            return False

        # Start backtracking
        # To avoid getting stuck in a specific bad path, we can randomize the start
        # or try multiple times.
        # Let's try a few random restarts with different value orderings.
        
        for attempt in range(10):
            # Clear rows
            for r in rows:
                r.clear()
            
            # We can randomize the order of rows? No, they are symmetric.
            # We can randomize the value selection?
            # Let's just run it. If it fails, try next M.
            
            try:
                if bt(0, 0, 0):
                    return [list(r) for r in rows]
            except TimeoutError:
                return None
        
        return None

    # Main loop
    best_solution = None
    current_M = lower_bound
    
    # Upper bound for M
    # Best known is 134. Let's go up to 150.
    max_M = lower_bound + 50
    
    while current_M <= max_M and time.time() < deadline - 10.0:
        sol = solve_for_scope(current_M)
        if sol is not None:
            best_solution = sol
            break
        current_M += 1
    
    if best_solution is None:
        # Fallback: Greedy construction with large scope
        # This ensures we return *something* valid, even if not optimal
        rows = []
        used = 0
        for i in range(n_rows):
            row = [0]
            # Greedy fill
            for j in range(1, k+1):
                # Find smallest v > row[-1] such that no conflict
                found = False
                for v in range(row[-1] + 1, 500):
                    new_diffs = 0
                    conflict = False
                    for u in row:
                        d = v - u
                        if (used >> d) & 1:
                            conflict = True
                            break
                        if (new_diffs >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    if conflict:
                        continue
                    row.append(v)
                    used |= new_diffs
                    found = True
                    break
                if not found:
                    # Should not happen with large scope
                    row.append(row[-1] + 1)
            rows.append(row)
        best_solution = rows
        
    return best_solution
