import sys
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Build n_rows rows of k+1 increasing integers starting at 0,
    such that all positive within-row differences are globally distinct.
    Minimize the largest entry.
    """
    # We will search for a solution.
    # To minimize the largest entry, we can use an iterative deepening approach
    # or just rely on the backtracking to find a good solution quickly.
    # Given the small size (n=4, k=7), a direct backtracking search with
    # heuristic ordering should find the optimal or near-optimal solution fast.
    
    best_solution = None
    best_max_val = float('inf')
    
    # We define the search state
    # rows: list of completed rows
    # current_row: list of integers in the row currently being built
    # used_diffs: set of all differences used so far
    
    def solve(rows, current_row, used_diffs, current_max):
        nonlocal best_solution, best_max_val
        
        # Pruning: If current_max is already >= best_max_val, we can stop this branch
        # Note: current_max is the max element in the current_row so far.
        # The final max will be at least current_max.
        if current_max >= best_max_val:
            return False
            
        # Check if we are done
        if len(rows) == n_rows and len(current_row) == 0:
            # Found a solution
            if current_max < best_max_val:
                best_max_val = current_max
                best_solution = [list(r) for r in rows]
            return True
            
        # If current_row is full, start a new row
        if len(current_row) == k + 1:
            new_rows = rows + [current_row]
            # The max of the new row is current_row[-1]
            # We need to update current_max to be the max of all rows so far?
            # Actually, current_max in the recursion should track the max of ALL elements placed so far.
            # Let's redefine: current_max is the max element in the *entire* partial solution.
            
            # If we just finished a row, the next row starts at 0.
            # The max element of the new row will be determined later.
            # So we call solve with empty current_row.
            return solve(new_rows, [], used_diffs, current_max)
            
        # Determine the next element for current_row
        # If current_row is empty, next is 0
        # Else, next must be > current_row[-1]
        
        if len(current_row) == 0:
            candidates = [0]
        else:
            start = current_row[-1] + 1
            # Heuristic: Limit the search space for the next element.
            # The max element in the final row will be at least start + (remaining elements needed).
            # Also, the differences must fit.
            # We iterate candidates in increasing order.
            # To improve, we can limit the upper bound of candidates.
            # A loose upper bound: current_max + some slack?
            # Better: Just iterate upwards, but prune if we exceed best_max_val.
            
            # We need to pick x such that {x - a for a in current_row} are not in used_diffs.
            
            found = False
            # We search for x starting from start
            # We can cap the search for x if x >= best_max_val, because then current_max would exceed best.
            # But we don't know the final max yet.
            
            # Let's generate candidates up to a reasonable limit or until we find one.
            # To make it "Best-First" like, we could collect a few candidates and pick the best one,
            # but simple DFS with pruning is often faster in Python due to overhead.
            
            # Optimization: Pre-compute diffs for current_row? No, it's small.
            
            # Upper bound for x in this step:
            # If we have 'rem' elements left to add to this row, the minimum possible increase in max value is 'rem'.
            # So if start + rem > best_max_val, we can prune.
            rem = (k + 1) - len(current_row)
            
            # We will iterate x from start upwards.
            # To avoid infinite loops or huge searches, we rely on the global best_max_val pruning.
            # Initially best_max_val is inf, so we might search a lot.
            # We should set an initial upper bound.
            # The simple counting bound is 112. The known best is 134.
            # Let's start with a high bound and let it tighten.
            
            # To speed up the first find, we can use a greedy heuristic to get an initial solution.
            # But the recursive solve will do that if we order candidates well.
            
            # Let's try candidates in increasing order.
            # We can limit the number of candidates checked per node to avoid explosion?
            # No, backtracking needs to explore.
            
            # A crucial pruning:
            # The set of differences generated by the current row so far is D_current.
            # The new differences will be {x - a for a in current_row}.
            # These must be disjoint from used_diffs.
            
            # Let's just iterate x.
            # To prevent the search from going too wide, we can use a heuristic:
            # Prefer x that results in smaller differences?
            
            # For this specific problem size, a simple DFS with backtracking should work.
            # We will iterate x from start to best_max_val - 1 (if best_max_val is set).
            # If best_max_val is inf, we need a cap. Let's use 200 as a safe cap for the first pass.
            
            upper_bound = best_max_val if best_max_val != float('inf') else 200
            
            for x in range(start, upper_bound + 1):
                # Check feasibility
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # New used diffs
                new_used_diffs = used_diffs | new_diffs
                
                # New current row
                new_current_row = current_row + [x]
                
                # New current max
                new_current_max = max(current_max, x)
                
                # Recursive step
                if solve(rows, new_current_row, new_used_diffs, new_current_max):
                    return True
                    
            return False

    # Initial call
    # We start with no rows, empty current row, no used diffs, max 0.
    # However, the first element must be 0.
    # The logic in solve handles len(current_row)==0 by trying candidates=[0].
    
    # To improve performance, we can use a heuristic to order rows?
    # No, rows are indistinguishable in terms of constraints, only the set of diffs matters.
    
    # One issue: The recursion might be slow if it explores many branches.
    # We can add a heuristic to the candidate selection:
    # Instead of just increasing x, we can try x that minimizes the maximum new difference?
    # Or x that is "closest" to the previous element?
    
    # Let's stick to increasing x, as it tends to find smaller solutions first.
    
    solve([], [], set(), 0)
    
    if best_solution:
        return best_solution
    else:
        # Fallback to greedy if search fails (should not happen for this size)
        # Re-implementing the seed's greedy as fallback
        used_diffs = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for _ in range(k):
                best_x = None
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        best_x = x
                        break
                    x += 1
                row.append(best_x)
                used_diffs |= {best_x - a for a in row[:-1]}
            rows.append(row)
        return rows
