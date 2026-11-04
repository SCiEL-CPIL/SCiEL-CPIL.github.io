import time
import sys

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 110.0
    
    # Best known result for (7,5) is 112. 
    # We will try to beat this or at least find a valid solution quickly.
    # Initial upper bound for scope
    best_scope = 112 
    best_result = None
    
    # Heuristic parameters
    lookahead_window = 20 # How far ahead to look for valid next elements
    
    def is_valid_row(row, used_diffs):
        """Check if a completed row is valid (no internal diff collisions with used_diffs)"""
        # This is implicitly checked during construction in the backtracker
        pass

    def count_valid_next(x, row, used_diffs, window):
        """
        Estimate how many valid next elements y exist in [x+1, x+window].
        y is valid if {y - a for a in row} are all in used_diffs (complement).
        """
        count = 0
        for y in range(x + 1, x + window + 1):
            new_diffs = {y - a for a in row}
            if not new_diffs & used_diffs:
                count += 1
        return count

    def backtrack(rows, used_diffs, row_idx, col_idx, current_scope_limit):
        """
        Recursive backtracking search.
        rows: list of completed rows
        used_diffs: set of all differences used so far
        row_idx: index of the row currently being built
        col_idx: index of the column (element) currently being chosen in the row
        current_scope_limit: max allowed value for the last element
        """
        nonlocal best_result, best_scope, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            # All rows constructed
            final_scope = max(r[-1] for r in rows)
            if final_scope < best_scope:
                best_scope = final_scope
                best_result = [r[:] for r in rows]
            return True
        
        if col_idx == 0:
            # Start a new row
            rows.append([0])
            return backtrack(rows, used_diffs, row_idx, 1, current_scope_limit)
        
        if col_idx == k + 1:
            # Row completed
            # Update scope limit for subsequent rows if this row's max is smaller
            # Actually, we just proceed to next row
            return backtrack(rows, used_diffs, row_idx + 1, 0, current_scope_limit)
        
        # Choose next element x
        last_val = rows[row_idx][-1]
        # Upper bound for x: must be < current_scope_limit
        # Also, we need to leave room for k+1-col_idx elements
        # Simple bound: x <= current_scope_limit - (k + 1 - col_idx)
        # But differences must be positive, so at least 1 apart.
        max_x = current_scope_limit - (k + 1 - col_idx)
        
        candidates = []
        
        for x in range(last_val + 1, max_x + 1):
            new_diffs = {x - a for a in rows[row_idx]}
            if new_diffs & used_diffs:
                continue
            
            # Heuristic: Count valid next elements in a window
            # This is expensive, so we only do it for a subset or optimize
            # To save time, we can just prioritize smaller x if we don't do lookahead.
            # But the directive asks for the CSP lookahead heuristic.
            
            # Optimization: Only do lookahead if we have time
            if time.time() - start_time > time_limit - 5:
                lookahead_score = 0
            else:
                lookahead_score = count_valid_next(x, rows[row_idx] + [x], used_diffs, lookahead_window)
            
            # Score: We want to maximize lookahead_score (more options).
            # Tie-breaker: Minimize x (smaller scope).
            # So sort by (-lookahead_score, x)
            candidates.append((-lookahead_score, x, new_diffs))
            
            # Early termination? If we find a candidate with high lookahead and small x, 
            # we might still need to check others to be sure.
            # Given the small size (k=5), the range is small.
        
        if not candidates:
            return False
        
        # Sort candidates
        candidates.sort(key=lambda item: (item[0], item[1]))
        
        # Try candidates in order
        for _, x, new_diffs in candidates:
            # Add x to current row
            rows[row_idx].append(x)
            # Update used diffs
            used_diffs.update(new_diffs)
            
            # Recurse
            if backtrack(rows, used_diffs, row_idx, col_idx + 1, current_scope_limit):
                # Check if we found a better solution
                if time.time() - start_time > time_limit:
                    return True # Stop searching, return best found
                if best_scope <= 105: # Theoretical lower bound or close
                    return True
                # Continue searching to see if we can find an even smaller scope?
                # The problem asks to minimize. 
                # If we found a solution, we can try to improve it.
                # But backtracking naturally explores the tree.
                # To improve scope, we need to reduce the upper bound.
                # Let's just return True if we found *any* valid solution with scope < best_scope?
                # No, backtrack returns True if it found a solution.
                # We should continue to explore if we want the *minimum*.
                # However, with a time limit, we might just return the first good one.
                # The heuristic "maximize options" tends to find solutions quickly.
                # Let's return True to stop the search once we have a valid solution,
                # unless we are in a mode to optimize. 
                # Given the 120s limit, finding *a* solution with scope <= 112 is good.
                # Finding < 112 is better.
                # Let's try to optimize: If we found a solution, we can restart with a tighter bound?
                # Or just let the recursion continue? 
                # If we return True, the parent calls return True, stopping the search.
                # This finds *a* solution. To find the *best*, we need to explore more.
                # Strategy: If we find a solution, record it. Then, if time remains, 
                # restart with best_scope - 1 as the limit.
                
                # For this implementation, let's just return the first valid solution found by this search.
                # The heuristic is designed to find low-scope solutions.
                return True
            
            # Backtrack
            rows[row_idx].pop()
            used_diffs.difference_update(new_diffs)
        
        return False

    # Start with a generous scope limit
    # We know 112 is possible. Let's start with 112 and try to go lower.
    # Or start with a high limit and let the heuristic find a low one?
    # The heuristic prefers small x, so it will naturally find small scopes.
    # Let's set the initial limit to 112.
    
    # To ensure we don't miss better solutions, we can run the search multiple times
    # with decreasing scope limits.
    
    current_limit = 112
    while current_limit > 105 and time.time() - start_time < time_limit:
        # Try to find a solution with scope <= current_limit
        # We use a fresh state for each limit
        # Note: The backtracker above modifies rows/used_diffs in place. 
        # We need to pass copies or reset them.
        
        # Let's wrap the backtrack call in a helper that manages state
        def try_solve(limit):
            rows = []
            used_diffs = set()
            # The backtrack function defined above uses nonlocal best_result.
            # It also modifies rows/used_diffs passed to it.
            # We need to be careful with state.
            
            # Let's re-implement the backtracker to be cleaner with state passing
            # or just use the one above but ensure we pass fresh lists.
            
            # The existing backtrack uses `rows` and `used_diffs` from the closure.
            # This is dangerous if we call it multiple times.
            # Let's refactor to pass state explicitly or reset.
            
            # For simplicity in this edit, I will replace the entire function logic 
            # to use a local backtracker that doesn't rely on nonlocal state mutation 
            # in a way that persists incorrectly, or I will just call it once.
            
            # Given the complexity, let's just run the search once with limit 112.
            # If it finds a solution < 112, great.
            # If it finds 112, we can try 111, etc.
            
            pass

    # Simpler approach:
    # Run the backtracking search. 
    # The backtracking search will find *a* solution. 
    # Due to the heuristic (prefer small x), it will likely find a low-scope solution.
    # We will run it with an initial limit of 112.
    # If it succeeds, we return the result.
    # If it fails (no solution <= 112), we return the fallback.
    
    # Let's refine the backtracker to be self-contained for a single run.
    
    def solve_with_limit(limit):
        rows = []
        used_diffs = set()
        
        def bt(row_idx, col_idx):
            nonlocal rows, used_diffs
            
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if col_idx == 0:
                rows.append([0])
                return bt(row_idx, 1)
            
            if col_idx == k + 1:
                return bt(row_idx + 1, 0)
            
            last_val = rows[row_idx][-1]
            max_x = limit - (k + 1 - col_idx)
            
            candidates = []
            for x in range(last_val + 1, max_x + 1):
                new_diffs = {x - a for a in rows[row_idx]}
                if new_diffs & used_diffs:
                    continue
                
                # Heuristic
                lookahead_score = 0
                if time.time() - start_time < time_limit - 10:
                    lookahead_score = count_valid_next(x, rows[row_idx] + [x], used_diffs, lookahead_window)
                
                candidates.append((-lookahead_score, x, new_diffs))
            
            if not candidates:
                return False
            
            candidates.sort(key=lambda item: (item[0], item[1]))
            
            for _, x, new_diffs in candidates:
                rows[row_idx].append(x)
                used_diffs.update(new_diffs)
                
                if bt(row_idx, col_idx + 1):
                    return True
                
                rows[row_idx].pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        if bt(0, 0):
            return [r[:] for r in rows]
        return None

    # Try to find the best solution by decreasing the limit
    # Start from 112 down to 105
    for limit in range(112, 104, -1):
        if time.time() - start_time > time_limit - 5:
            break
            
        result = solve_with_limit(limit)
        if result:
            best_result = result
            best_scope = limit
            break # Found the best possible for this limit, stop? 
            # Actually, if we found a solution for limit L, there might be one for L-1.
            # But we are iterating downwards. 
            # If we found one for 112, we should try 111.
            # The loop above does this: it tries 112, if success, breaks? 
            # No, if I break, I stop. 
            # I should NOT break if I want to find the *minimum*.
            # But `solve_with_limit` finds *a* solution.
            # If I find a solution for 112, I should try 111.
            # So I should NOT break. I should continue the loop.
            # But `best_result` should be updated to the *lowest* limit that works.
            
    # Correcting the loop logic:
    best_found = None
    for limit in range(112, 104, -1):
        if time.time() - start_time > time_limit - 5:
            break
        
        result = solve_with_limit(limit)
        if result:
            best_found = result
            # Don't break, try smaller limits
            # But if we found 105, we can break as it's the theoretical min
            if limit <= 105:
                break
    
    if best_found:
        return best_found
    
    # Fallback to greedy if backtracking failed or timed out
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
