import time
import itertools

def construct(n_rows, k):
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    # Calculate lower bound for scope
    # Total differences needed: n_rows * (k+1)*k/2
    # Differences are distinct integers from 1 to M
    # So M >= total_diffs
    total_diffs_needed = n_rows * (k + 1) * k // 2
    min_scope = total_diffs_needed
    
    # Initial greedy solution to get a valid upper bound
    def greedy_construct():
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
        return rows

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)
    
    # Try to improve the scope using backtracking
    # We try scopes from min_scope up to best_scope - 1
    # If we find a solution for scope M, we update best and continue to try smaller?
    # No, we want the MINIMUM scope. So we should try from min_scope upwards.
    # If we find a valid solution for M, that is the answer (since we are going up).
    # But backtracking might be slow for small M if no solution exists.
    # Strategy: 
    # 1. Try M = min_scope.
    # 2. If no solution found within a time slice (e.g., 5s), try M = min_scope + 1.
    # 3. Keep track of the best valid solution found (which might be from a larger M if we failed to prove smaller M impossible? No, if we find a solution for M, it's valid. We want the smallest M).
    # Actually, if we find a solution for M=105, we return it.
    # If we don't find one for 105, we try 106.
    # If we run out of time, we return the best valid solution we *have* (which is the greedy one, or a solution for a larger M if we had found one).
    # Wait, if we try M=105 and fail, we don't have a solution for 105. We still have the greedy solution for M=best_scope.
    # So we should only return a solution for M if we *found* one.
    
    # Let's refine: 
    # We will search for a solution with scope exactly M.
    # If found, return it immediately (since we search M in increasing order).
    # If not found within time, we move to M+1.
    # If we run out of total time, we return the initial greedy solution.
    
    def solve_for_scope(M):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row, last_val):
            nonlocal used_diffs, rows
            
            # Time check
            if time.time() - start_time > timeout:
                return False
            
            # Base case: all rows constructed
            if row_idx == n_rows:
                return True
            
            # If starting a new row
            if len(current_row) == 0:
                # Symmetry breaking: new row must be lexicographically >= previous row
                # Since we build rows in order, and 0 is always first,
                # we can enforce that the second element of this row is >= second element of previous row?
                # Actually, simple lexicographic sort of the final rows is needed.
                # To enforce this during construction:
                # The tuple current_row (when complete) must be >= rows[-1] (if any).
                # It's easier to just generate all and check at the end, but that's expensive.
                # Heuristic: force the first non-zero element of row i to be >= first non-zero of row i-1.
                # This is a valid symmetry break for DTS? Yes, rows are distinct.
                
                if rows:
                    prev_row = rows[-1]
                    # We will build the row. The first element is 0.
                    # The second element (index 1) must be >= prev_row[1]?
                    # Not strictly, because the rows are sets of differences.
                    # But lexicographically, if row A < row B, we can swap them.
                    # So we can assume Row 1 <= Row 2 <= ... in lexicographic order.
                    # So current_row[1] must be >= prev_row[1].
                    # And if current_row[1] == prev_row[1], then current_row[2] >= prev_row[2], etc.
                    
                    # Implementation:
                    # We don't know current_row[1] yet.
                    # We can pass a lower bound for the next element.
                    pass 
                
                # Start new row with 0
                current_row = [0]
                last_val = 0
                # Recurse to add next element
                return backtrack(row_idx, current_row, last_val)
            
            # If current row is full
            if len(current_row) == k + 1:
                # Check symmetry: this row must be >= previous row
                if rows:
                    if current_row < rows[-1]:
                        return False
                
                rows.append(current_row)
                # Recurse to next row
                return backtrack(row_idx + 1, [], 0)
            
            # Determine next element candidates
            # We need to pick x such that x > last_val and x <= M
            # And diffs x - y for y in current_row are not in used_diffs
            
            # Optimization: 
            # If we are on the last row, we can be more aggressive?
            
            # Lower bound for x:
            # If symmetry breaking is active:
            lower_bound = last_val + 1
            if rows and len(current_row) < k + 1:
                # Compare with previous row
                prev_row = rows[-1]
                # We need current_row >= prev_row
                # We have built current_row[0..len-1].
                # If current_row[0..len-1] == prev_row[0..len-1], then next x must be >= prev_row[len]
                # If current_row[0..len-1] > prev_row[0..len-1], then x can be anything > last_val
                # If current_row[0..len-1] < prev_row[0..len-1], then it's invalid (should have been pruned earlier?)
                # Actually, if at any prefix current_row < prev_row, the whole row will be < prev_row.
                # So we can prune if current_row < prev_row at any prefix.
                
                # Check if current_row (prefix) is already < prev_row (prefix)
                # Since we build sequentially:
                # If len(current_row) > 0:
                #   if current_row < prev_row[:len(current_row)]: return False (prune)
                #   if current_row == prev_row[:len(current_row)]:
                #       lower_bound = max(lower_bound, prev_row[len(current_row)])
                
                if len(current_row) > 0:
                    if current_row < prev_row[:len(current_row)]:
                        return False
                    if current_row == prev_row[:len(current_row)]:
                        lower_bound = max(lower_bound, prev_row[len(current_row)])
            
            # Iterate possible x
            # Heuristic: try smaller x first to keep scope tight?
            # But we are fixing M. So we just need *any* x.
            # Trying smaller x might fill up small differences, potentially blocking later rows?
            # Usually, packing small differences is good.
            
            for x in range(lower_bound, M + 1):
                new_diffs = {x - y for y in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # Check if we can still complete the row
                # Remaining slots: k + 1 - (len(current_row) + 1)
                # This is a weak check.
                
                # Try this x
                new_used = used_diffs | new_diffs
                new_row = current_row + [x]
                
                if backtrack(row_idx, new_row, x):
                    # Success
                    # Note: used_diffs and rows are nonlocal and modified in place?
                    # I used nonlocal rows, but used_diffs is a set. Sets are mutable.
                    # But I passed new_used. I need to update the nonlocal used_diffs.
                    used_diffs = new_used
                    return True
                else:
                    # Backtrack: do nothing to used_diffs since we didn't modify it in place 
                    # (we created new_used). But wait, I need to restore used_diffs if I modify it in place.
                    # In my code above: `used_diffs = new_used` inside the success branch.
                    # If it fails, used_diffs remains as it was.
                    # This is correct.
                    pass
            
            return False
        
        return backtrack(0, [], 0)

    # Search loop
    current_M = min_scope
    while current_M < best_scope:
        if time.time() - start_time > timeout:
            break
        
        # Try to find a solution for current_M
        # We need to reset state for each M? Yes.
        # The solve_for_scope function uses local variables, so it's safe.
        
        # Optimization: If M is very close to min_scope, it might be infeasible.
        # We just try it.
        
        if solve_for_scope(current_M):
            # We found a solution for current_M
            # Reconstruct the rows? 
            # The solve_for_scope modifies `rows` inside its closure? 
            # No, `rows` is local to solve_for_scope.
            # I need to return the rows.
            
            # Let's refactor solve_for_scope to return the rows if successful.
            pass
            
        current_M += 1
        
    # If we didn't find a better solution, return the greedy one
    return best_rows

# The logic above has a flaw: solve_for_scope doesn't return the rows.
# And the symmetry breaking logic needs to be careful.
# Also, the time check inside backtrack is expensive if done at every node.
# We should check time every N nodes or use a global counter.

# Let's rewrite more carefully.
