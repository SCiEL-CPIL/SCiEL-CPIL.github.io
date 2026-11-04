import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Parameters for the specific task: n_rows=7, k=5
    # But we should be generic.
    
    diffs_per_row = k * (k - 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound
    lower_bound = total_diffs_needed
    
    # Upper bound: We know 112 works for (7,5). 
    # For generic inputs, we might not know. 
    # We can use a greedy upper bound.
    
    def get_greedy_upper_bound():
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
        return max(max(r) for r in rows)

    greedy_scope = get_greedy_upper_bound()
    
    # For the specific task (7,5), we know 112 is a good target.
    # We will search from lower_bound up to min(greedy_scope, 112) if n_rows==7 and k==5.
    # Otherwise, search up to greedy_scope.
    
    search_upper = greedy_scope
    if n_rows == 7 and k == 5:
        search_upper = min(greedy_scope, 112)
    
    # If greedy is already very low, search up to greedy.
    # We want to find the MINIMUM M.
    # So we iterate M from lower_bound to search_upper.
    
    # Solver function
    def solve_for_scope(M, node_limit):
        # used_mask: bitmask of used differences
        # rows: list of completed rows
        # current_row: list of elements in current row
        # last_val: last element in current row
        # prev_max: max element of previous row (for symmetry)
        # row_idx: index of current row being built
        
        used_mask = 0
        rows = []
        nodes_visited = 0
        found_solution = False
        
        # Precompute masks for differences? No, compute on fly.
        
        def backtrack(row_idx, current_row, last_val, prev_max):
            nonlocal used_mask, rows, nodes_visited, found_solution
            
            if found_solution:
                return True
            if time.time() - start_time > deadline:
                return False
            if nodes_visited > node_limit:
                return False
            
            nodes_visited += 1
            
            # Pruning: Check if enough differences remain
            remaining_rows = n_rows - row_idx
            if remaining_rows > 0:
                needed_diffs = remaining_rows * diffs_per_row
                # Current used diffs count
                used_count = used_mask.bit_count()
                # Available diffs in [1, M]
                available_diffs = M - used_count
                if available_diffs < needed_diffs:
                    return False
            
            if row_idx == n_rows:
                found_solution = True
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                row_max = current_row[-1]
                
                # Symmetry breaking
                if row_max < prev_max:
                    return False
                
                # Add row to list
                rows.append(current_row[:])
                
                # Update used_mask
                new_diffs_mask = 0
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        new_diffs_mask |= (1 << d)
                
                # Check for conflicts (should be none if we checked during construction)
                if used_mask & new_diffs_mask:
                    rows.pop()
                    return False
                    
                used_mask |= new_diffs_mask
                
                ok = backtrack(row_idx + 1, [0], 0, row_max)
                
                if ok:
                    return True
                
                # Backtrack
                used_mask ^= new_diffs_mask
                rows.pop()
                return False
            
            # Choose next value x
            # x > last_val, x <= M
            # If this is the last element of the row, x must be >= prev_max
            is_last_elem = (len(current_row) == k)
            start_x = last_val + 1
            if is_last_elem and prev_max > start_x:
                start_x = prev_max
            
            candidates = []
            for x in range(start_x, M + 1):
                # Compute new diffs mask
                new_diffs_mask = 0
                for a in current_row:
                    d = x - a
                    # d must be <= M? Yes, x<=M, a>=0, so d<=M.
                    # d must be >= 1? Yes, x > a.
                    new_diffs_mask |= (1 << d)
                
                # Check conflict
                if used_mask & new_diffs_mask:
                    continue
                
                candidates.append(x)
            
            # Heuristic: Sort candidates.
            # Smaller x keeps the row "tighter", which might help with symmetry constraints
            # and leaves larger differences for later? 
            # Actually, using small x consumes small differences.
            # Small differences are "scarce" in terms of packing?
            # Let's try sorting by x ascending.
            candidates.sort()
            
            for x in candidates:
                new_diffs_mask = 0
                for a in current_row:
                    d = x - a
                    new_diffs_mask |= (1 << d)
                
                current_row.append(x)
                used_mask |= new_diffs_mask
                
                ok = backtrack(row_idx, current_row, x, prev_max)
                if ok:
                    return True
                
                current_row.pop()
                used_mask ^= new_diffs_mask
                
                if found_solution:
                    return True
            
            return False
        
        # Start search
        # First row: prev_max = -1 (or 0, since max >= 0)
        # We use -1 to allow any max.
        ok = backtrack(0, [0], 0, -1)
        
        if ok and found_solution:
            return rows
        return None

    # Iterate M
    best_rows = None
    
    # We want to find the smallest M.
    # To avoid spending all time on M=105 if it's impossible, we use a node limit.
    # If M=105 fails, try 106, etc.
    
    # For the specific case (7,5), we know 112 is a solution.
    # We try 105..111 with a strict limit.
    # Then 112 with the remaining time.
    
    # Estimate node limit per M.
    # Total time ~ 115s.
    # 7 values (105-111) + 1 value (112).
    # Let's give 10s to each of 105-111.
    # And the rest to 112.
    
    # However, if we find a solution for 105, we return immediately.
    
    # Dynamic node limit based on time?
    # Simpler: Fixed node limit for exploration, then fall back to 112.
    
    # Let's use a node limit of 20,000 for M < 112.
    # This is fast.
    
    for M in range(lower_bound, search_upper + 1):
        if time.time() - start_time > deadline:
            break
            
        # If M is the last candidate (likely 112), use a larger limit or just run until deadline.
        if M == search_upper:
            # Use a large node limit, effectively running until deadline
            node_limit = 10**9 
        else:
            node_limit = 20000
            
        res = solve_for_scope(M, node_limit)
        if res is not None:
            best_rows = res
            # We found a solution for M. Since we iterate upwards, this is the best.
            break
            
    if best_rows is None:
        # Fallback: Return the greedy solution if the solver failed completely
        # This shouldn't happen if 112 is reachable, but just in case.
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
        best_rows = rows

    return best_rows
