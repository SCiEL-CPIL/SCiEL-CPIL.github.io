"""Rewritten program using backtracking with constraint propagation for difference triangle set."""
import sys
import time
import random

# Increase recursion limit for deep backtracking if needed
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all pairwise differences
    within rows are distinct across all rows.
    
    Uses backtracking with forward checking (constraint propagation).
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * n_rows] if k == 0 else []
    
    # Total number of differences needed: n_rows * C(k+1, 2)
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # The minimum possible scope is total_diffs (counting bound).
    # We search for the smallest M such that a valid configuration exists with max entry <= M.
    
    # Heuristic lower bound:
    # The differences must be distinct and <= M.
    # There are total_diffs differences.
    # So M must be at least total_diffs.
    lower_bound = total_diffs
    
    # Upper bound heuristic:
    # A greedy construction usually provides a decent upper bound.
    # Let's generate a greedy solution first to set a tight upper bound.
    def greedy_construct(n_rows, k):
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
        return max(max(r) for r in rows)
    
    upper_bound = greedy_construct(n_rows, k)
    
    # Search for the optimal M in [lower_bound, upper_bound]
    # We can use binary search if we had a decision procedure, but backtracking is often faster
    # for small M. We'll iterate M upwards from lower_bound.
    # To be safe against timeouts, we limit the search time.
    
    start_time = time.time()
    timeout = 100.0  # Leave 20s buffer for overhead
    
    # Pre-compute possible differences? No, dynamic.
    
    # Backtracking solver
    # State: 
    #   rows: list of current rows (partially filled)
    #   used_diffs: set of all differences used so far
    #   current_row_idx: which row we are filling
    #   current_pos_in_row: which position in the current row we are filling
    #   current_row: the list of values in the current row so far
    
    # Optimization: 
    # 1. Fill rows one by one.
    # 2. Within a row, choose values in increasing order.
    # 3. Forward checking: when adding a value, check all new differences against used_diffs and within the current row.
    
    def solve(n_rows, k, max_val):
        """
        Try to find a valid configuration with all entries <= max_val.
        Returns list of rows if found, else None.
        """
        # Sort rows to fill? 
        # Usually, filling "harder" rows first is better, but for DTS, symmetry suggests order doesn't matter much 
        # unless we use symmetry breaking. 
        # Symmetry breaking: 
        # 1. First row must be [0, ...]
        # 2. We can assume the first row is the one with the smallest max value? 
        #    Actually, for DTS, the rows are not ordered by size necessarily in the optimal solution, 
        #    but we can fix the first row to reduce symmetry? 
        #    Let's just fill rows 0 to n_rows-1.
        
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, pos_in_row, current_row):
            # Check time
            if time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                # All rows filled
                return True
            
            if pos_in_row == k + 1:
                # Current row is complete
                rows.append(current_row)
                return backtrack(row_idx + 1, 0, [0])
            
            # Determine the range for the next value
            # If pos_in_row == 0, value is 0
            if pos_in_row == 0:
                val = 0
                new_diffs = set()
                # No diffs with 0-0
                # Check if 0 is valid (it always is, as it's the start)
                # But we must ensure subsequent values can be formed.
                # Recurse
                return backtrack(row_idx, pos_in_row + 1, [0])
            
            # Determine lower bound for val
            if pos_in_row == 1:
                # Second element in row
                # Must be > 0
                # Also, the difference val - 0 = val must not be in used_diffs
                lo = 1
            else:
                lo = current_row[-1] + 1
            
            # Determine upper bound for val
            hi = max_val
            
            # Heuristic: try smaller values first to keep scope small? 
            # Or try values that allow more room for later?
            # Usually, trying smaller values first is good for finding *a* solution within max_val.
            
            # Optimization: 
            # The differences generated by adding val to current_row are:
            # val - current_row[i] for i in range(pos_in_row)
            # These must be:
            # 1. Distinct from each other (guaranteed if current_row is strictly increasing)
            # 2. Distinct from used_diffs
            # 3. Distinct from differences already in current_row (which are val - current_row[i] vs current_row[j] - current_row[i]?)
            #    Wait, the differences *within* the current row are:
            #    existing: current_row[j] - current_row[i] for 0 <= i < j < pos_in_row
            #    new: val - current_row[i] for 0 <= i < pos_in_row
            #    We need new diffs to be distinct from existing diffs in current_row AND used_diffs.
            
            # Let's pre-calculate the set of diffs already in current_row
            # This is O(k^2) per step, which is fine for small k.
            existing_row_diffs = set()
            for i in range(pos_in_row):
                for j in range(i + 1, pos_in_row):
                    existing_row_diffs.add(current_row[j] - current_row[i])
            
            # Iterate candidate values
            # To improve speed, we can skip values that immediately cause a conflict.
            # Conflict if (val - current_row[i]) in (used_diffs | existing_row_diffs)
            
            # Note: used_diffs contains diffs from *previous* rows.
            # existing_row_diffs contains diffs from *current* row so far.
            # The new diffs are val - current_row[i].
            
            # We iterate val from lo to hi.
            # For better performance, we can randomize the order if we fail often?
            # But for finding the *smallest* max_val, we are iterating M upwards.
            # Within a fixed M, we just need *any* solution.
            # Randomization can help escape local dead-ends caused by row ordering.
            
            # Let's try a simple loop first. If it's too slow, we add randomization.
            
            # Optimization: If k is small, we can just loop.
            
            # To break symmetry between rows:
            # We can assume the first row is "canonical". 
            # But for now, let's just search.
            
            # One key optimization: 
            # The set of available differences is {1, ..., max_val} minus used_diffs.
            # We need to pick k differences for the new row that form a "difference set" relative to 0?
            # No, the row is [0, x1, x2, ..., xk].
            # The diffs are x1, x2, ..., xk, x2-x1, x3-x1, ..., xk-x1, x3-x2, ..., xk-x2, ..., xk-x_{k-1}.
            
            # Let's just iterate val.
            
            # To speed up, we can check feasibility of the remaining positions?
            # For now, simple backtracking.
            
            # We'll iterate through possible values.
            # To make it robust, we'll shuffle the candidate values if we suspect we're stuck,
            # but since we want to find *a* solution for a given M, and M is small, 
            # deterministic order might be fine.
            
            # However, to avoid worst-case exponential blowup, we can use a heuristic:
            # Try values that are "spread out"?
            
            # Let's stick to sequential for now, but add a time check.
            
            # We can prune: if the number of remaining differences needed exceeds the number of available differences <= max_val, fail.
            # Remaining diffs needed for this row: 
            # Total diffs in a row of size k+1 is k(k+1)/2.
            # Diffs already in current_row: pos_in_row*(pos_in_row-1)/2.
            # Diffs to be added in this row: total_row_diffs - existing_row_diff_count.
            # But we are adding them one by one.
            
            # Let's just iterate.
            
            # To improve performance, we can use a "conflict-driven" approach:
            # Instead of trying all val, we can look at the constraints.
            # But for k=5, the branching factor is manageable.
            
            # Let's iterate val from lo to hi.
            # We can speed this up by pre-computing which values are "bad" due to used_diffs.
            # A value val is bad if any (val - current_row[i]) is in used_diffs or existing_row_diffs.
            
            # Let's build a set of "forbidden" vals.
            # Forbidden if:
            # 1. val - current_row[i] in used_diffs for any i
            # 2. val - current_row[i] in existing_row_diffs for any i
            
            # This is O(k * |used_diffs|) which might be slow if used_diffs is large.
            # But |used_diffs| <= total_diffs.
            
            # Alternative: 
            # For each diff d in used_diffs, val = d + current_row[i] is forbidden.
            # For each diff d in existing_row_diffs, val = d + current_row[i] is forbidden.
            
            # Let's generate the set of forbidden values.
            forbidden = set()
            for i in range(pos_in_row):
                ci = current_row[i]
                for d in used_diffs:
                    forbidden.add(d + ci)
                for d in existing_row_diffs:
                    forbidden.add(d + ci)
            
            # Now iterate val in [lo, hi] not in forbidden.
            # We can also check if val - current_row[i] creates a conflict *within* the new diffs?
            # The new diffs are val - current_row[i].
            # Are they distinct? Yes, because current_row is strictly increasing.
            # Do they conflict with each other? No, distinct.
            # Do they conflict with existing_row_diffs? Checked via forbidden.
            # Do they conflict with used_diffs? Checked via forbidden.
            
            # So if val is not in forbidden, it is a valid candidate.
            
            # We iterate val from lo to hi.
            # To add some randomization to escape bad paths, we can pick randomly from valid candidates?
            # No, we want to find *a* solution. Sequential is fine.
            
            # But wait, if we just pick the first valid val, we might paint ourselves into a corner for the next positions.
            # Backtracking handles this by trying other vals if the next step fails.
            
            # Optimization: If there are too many candidates, we might want to be smart.
            # But for small k, it's fine.
            
            # Let's iterate.
            # To avoid iterating through thousands of values, we can just loop.
            
            # One more check: 
            # The maximum value in the row will be at most max_val.
            # The remaining positions must be filled with values > val and <= max_val.
            # This is implicitly handled by the recursion bounds.
            
            # Let's try candidates in increasing order.
            for val in range(lo, hi + 1):
                if val in forbidden:
                    continue
                
                # Check if val is strictly greater than current_row[-1]
                if val <= current_row[-1]:
                    continue
                
                # Add val
                new_row = current_row + [val]
                
                # Compute new diffs
                new_diffs = set()
                valid = True
                for i in range(pos_in_row):
                    d = val - current_row[i]
                    # We already checked against used_diffs and existing_row_diffs via 'forbidden'
                    # But we should add them to the set of diffs for this row for the next step?
                    # No, the 'forbidden' set was computed for *this* step.
                    # In the next step, these new diffs will be part of 'existing_row_diffs'.
                    
                    # Wait, my 'forbidden' logic checks if `d` is in `used_diffs` or `existing_row_diffs`.
                    # It does NOT check if `d` conflicts with *other* new diffs?
                    # The new diffs are `val - current_row[i]`.
                    # Are `val - current_row[i]` and `val - current_row[j]` distinct?
                    # Yes, because `current_row[i] != current_row[j]`.
                    # So no internal conflict among new diffs.
                    
                    new_diffs.add(d)
                
                # The new set of diffs for the current row will be `existing_row_diffs | new_diffs`.
                # We don't need to update `used_diffs` yet, because `used_diffs` is global for *completed* rows.
                # The `existing_row_diffs` is local to the current row.
                
                # Recurse
                if backtrack(row_idx, pos_in_row + 1, new_row):
                    return True
            
            return False
        
        # Start backtracking
        # To break symmetry, we can fix the first row? 
        # No, let's just run.
        
        # One issue: `used_diffs` is passed by reference? 
        # In Python, sets are mutable. 
        # We need to update `used_diffs` when a row is completed.
        # But `used_diffs` is defined in the outer scope of `solve`.
        # We need to pass it or modify it carefully.
        # Since we are backtracking, we need to "undo" changes to `used_diffs`.
        
        # Let's refactor `solve` to handle `used_diffs` properly.
        
        def solve_inner(n_rows, k, max_val):
            used_diffs = set()
            rows = []
            
            def backtrack(row_idx, pos_in_row, current_row):
                if time.time() - start_time > timeout:
                    return False
                
                if row_idx == n_rows:
                    return True
                
                if pos_in_row == k + 1:
                    # Row complete
                    # Add all diffs of this row to used_diffs
                    row_diffs = set()
                    for i in range(len(current_row)):
                        for j in range(i + 1, len(current_row)):
                            row_diffs.add(current_row[j] - current_row[i])
                    
                    # Check for conflict? 
                    # The backtracking step ensured no conflict with *previous* used_diffs.
                    # But did it ensure no conflict *within* the row?
                    # Yes, `forbidden` included `existing_row_diffs`.
                    # So `row_diffs` is disjoint from `used_diffs`.
                    
                    # Save row
                    rows.append(current_row)
                    # Update used_diffs
                    used_diffs.update(row_diffs)
                    
                    if backtrack(row_idx + 1, 0, [0]):
                        return True
                    
                    # Undo
                    rows.pop()
                    used_diffs.difference_update(row_diffs)
                    return False
                
                # Determine lo, hi
                if pos_in_row == 0:
                    return backtrack(row_idx, 1, [0])
                
                lo = current_row[-1] + 1
                hi = max_val
                
                existing_row_diffs = set()
                for i in range(pos_in_row):
                    for j in range(i + 1, pos_in_row):
                        existing_row_diffs.add(current_row[j] - current_row[i])
                
                forbidden = set()
                for i in range(pos_in_row):
                    ci = current_row[i]
                    for d in used_diffs:
                        forbidden.add(d + ci)
                    for d in existing_row_diffs:
                        forbidden.add(d + ci)
                
                for val in range(lo, hi + 1):
                    if val in forbidden:
                        continue
                    if val <= current_row[-1]:
                        continue
                    
                    new_row = current_row + [val]
                    if backtrack(row_idx, pos_in_row + 1, new_row):
                        return True
                
                return False
            
            if backtrack(0, 0, [0]):
                return rows
            return None
        
        return solve_inner(n_rows, k, max_val)
    
    # Search for the smallest M
    # Start from lower_bound
    # If lower_bound is too tight, increment.
    
    # To speed up, we can jump. But for small problems, linear search is fine.
    # We'll add a randomization to the row filling order if we get stuck?
    # No, backtracking is exhaustive for a fixed M.
    
    # One improvement: 
    # If the search for M takes too long, we might want to try M+1.
    # But the timeout is global.
    
    # Let's try M from lower_bound to upper_bound.
    # If we find a solution, return it.
    
    # To make it faster, we can randomize the order of rows?
    # No, the structure is fixed.
    
    # Let's run the search.
    # If it times out, we return the greedy solution (which is valid, just not optimal).
    
    best_solution = None
    for M in range(lower_bound, upper_bound + 1):
        if time.time() - start_time > timeout:
            break
        
        solution = solve(n_rows, k, M)
        if solution is not None:
            best_solution = solution
            break
    
    if best_solution is None:
        # Fallback to greedy
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
    
    return best_solution
