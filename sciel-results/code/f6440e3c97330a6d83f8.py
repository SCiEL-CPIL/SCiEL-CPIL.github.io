import time
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs n_rows rows of k+1 increasing integers starting at 0.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_solution = None
    best_scope = float('inf')
    
    # We will use backtracking to find a valid configuration.
    # To minimize scope, we prioritize smaller numbers.
    
    def solve(row_idx, current_row, used_diffs, current_max):
        nonlocal best_solution, best_scope
        
        # Check time
        if time.time() > deadline:
            return False
        
        # If all rows are filled, we have a solution
        if row_idx == n_rows:
            if current_max < best_scope:
                best_scope = current_max
                # Deep copy the current state (rows are built incrementally, 
                # but we need to store the full set of rows)
                # We need to pass the full list of rows to store the solution
                # Let's restructure to pass rows list
                pass # Handled in wrapper
            return True

        # If we are starting a new row, current_row is [0]
        # If we are extending a row, current_row has some elements
        
        # Optimization: If current_row is empty, it means we are starting a new row?
        # Let's define the state more clearly.
        
        # Actually, let's just build row by row.
        # State:
        # rows: list of completed rows
        # current_row: list of elements in the row currently being built
        # used_diffs: set of all differences used in completed rows AND current_row
        
        # This is getting complex to manage in a single recursive call without passing heavy structures.
        # Let's use a class or nested function with nonlocal variables for state.
        
        pass

    # Better structure:
    
    rows = []
    used_diffs = set()
    
    def backtrack(row_idx, current_row):
        nonlocal rows, used_diffs, best_solution, best_scope, deadline
        
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            # Check if this is a better solution
            # The scope is max(max(row) for row in rows)
            # Since we build rows sequentially and add to rows only when complete,
            # and we try smallest numbers first, the first complete solution is likely good.
            # But to be sure, we track best_scope.
            current_scope = max(max(r) for r in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_solution = [r[:] for r in rows]
            return True
            
        if not current_row:
            # Start new row
            current_row = [0]
            # 0 is always the start. No differences yet.
            # Recurse to add next element
            return backtrack(row_idx, current_row)
            
        if len(current_row) == k + 1:
            # Row is complete
            rows.append(current_row)
            # Calculate new used_diffs for this row
            new_diffs = set()
            for i in range(len(current_row)):
                for j in range(i+1, len(current_row)):
                    d = current_row[j] - current_row[i]
                    if d in used_diffs:
                        # This shouldn't happen if we checked during construction, 
                        # but it's a safety check.
                        rows.pop()
                        return False
                    new_diffs.add(d)
            
            # Check if new_diffs conflicts with global used_diffs
            # Actually, we should have checked conflicts when adding each element.
            # So used_diffs should already be updated?
            # Let's update used_diffs BEFORE recursing to next row.
            
            # To make backtracking easier, we should update used_diffs incrementally.
            # But here, we are adding the whole row at once.
            # This is less efficient for pruning.
            
            # Let's refactor to add elements one by one.
            return False # This path is not taken in the new design below

    # REFACTOR: Add elements one by one to allow better pruning
    
    rows = []
    used_diffs = set()
    
    def backtrack(row_idx, current_row, current_max):
        nonlocal rows, used_diffs, best_solution, best_scope, deadline
        
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            if current_max < best_scope:
                best_scope = current_max
                best_solution = [r[:] for r in rows]
            return True
            
        if not current_row:
            # Start new row with 0
            return backtrack(row_idx, [0], 0)
            
        if len(current_row) == k + 1:
            # Row complete. Commit it.
            rows.append(current_row)
            # The differences were already added to used_diffs during construction
            # So we just move to next row
            result = backtrack(row_idx + 1, [], current_max)
            if result:
                return True
            # Backtrack
            rows.pop()
            # We don't need to remove diffs from used_diffs if we managed it carefully?
            # No, we MUST remove them if we backtrack.
            # This makes it tricky.
            
            # Alternative: Pass used_diffs as a parameter (immutable copy or undo log).
            # Given the small size (112 diffs), copying the set is fast.
            # But we are in a deep recursion.
            
            # Let's try a different approach: 
            # We don't backtrack within a row easily if we modify global state.
            # Let's use a local set for the current row's diffs, and merge only when row is complete?
            # No, we need to check against *all* previous rows' diffs.
            
            # So:
            # Global used_diffs: diffs from rows 0 to row_idx-1
            # Local row_diffs: diffs from current_row
            # When adding x to current_row:
            #   new_diffs = {x - y for y in current_row}
            #   if new_diffs & (global_used_diffs | row_diffs): invalid
            #   else: add to row_diffs, recurse
            
            # This is clean.
            return False # Placeholder

    # Correct Implementation
    
    rows = []
    global_used_diffs = set()
    
    def backtrack(row_idx, current_row, row_diffs, current_max):
        nonlocal rows, global_used_diffs, best_solution, best_scope, deadline
        
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            if current_max < best_scope:
                best_scope = current_max
                best_solution = [r[:] for r in rows]
            return True
            
        if not current_row:
            # Start new row
            return backtrack(row_idx, [0], set(), current_max)
            
        if len(current_row) == k + 1:
            # Row is complete
            # Check if row_diffs conflicts with global_used_diffs
            # (This should have been checked incrementally, but let's be safe)
            if row_diffs & global_used_diffs:
                return False
                
            # Commit row
            rows.append(current_row)
            # Update global_used_diffs
            # We need to be able to undo this.
            # Since we are passing by reference, we must restore it on backtrack.
            
            new_global = global_used_diffs.union(row_diffs)
            # Recurse
            result = backtrack(row_idx + 1, [], set(), current_max)
            if result:
                return True
            # Backtrack
            rows.pop()
            # global_used_diffs is restored by the caller's context?
            # No, we modified it in place if we did `global_used_diffs |= row_diffs`.
            # Let's not modify in place. Let's pass the new set.
            # But that's expensive? 112 ints is small.
            
            # Let's change signature to pass global_used_diffs
            return False # Placeholder

    # Final Structure
    
    def solve_with_scope_limit(max_scope):
        """
        Attempts to find a solution where all entries <= max_scope.
        Returns the solution if found, else None.
        """
        rows = []
        
        def backtrack(row_idx, current_row, row_diffs, global_diffs):
            if time.time() > deadline:
                return None
                
            if row_idx == n_rows:
                return [r[:] for r in rows]
                
            if not current_row:
                # Start new row
                return backtrack(row_idx, [0], set(), global_diffs)
                
            if len(current_row) == k + 1:
                # Row complete
                if row_diffs & global_diffs:
                    return None
                    
                rows.append(current_row)
                new_global = global_diffs.union(row_diffs)
                
                result = backtrack(row_idx + 1, [], set(), new_global)
                if result:
                    return result
                    
                rows.pop()
                return None
                
            # Extend current row
            last_val = current_row[-1]
            # Try next values
            # Optimization: We can start from last_val + 1
            # We can also limit to max_scope
            for x in range(last_val + 1, max_scope + 1):
                # Check if x is valid
                new_diffs = set()
                valid = True
                for y in current_row:
                    d = x - y
                    if d in row_diffs or d in global_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                    
                if valid:
                    # Recurse
                    result = backtrack(row_idx, current_row + [x], row_diffs.union(new_diffs), global_diffs)
                    if result:
                        return result
            
            return None

        return backtrack(0, [], set(), set())

    # Strategy:
    # 1. Try to find a solution with a reasonably low scope.
    # 2. The counting bound is 112. The known best is 134.
    # 3. We can try increasing scopes until we find one.
    # 4. However, backtracking with "smallest first" naturally finds small scopes first.
    # 5. So we can just run backtrack with a high max_scope (e.g. 200) and rely on the search order.
    # 6. To ensure we don't miss a smaller scope due to time limits, we can't easily do that with a single run.
    # 7. But the greedy nature of "smallest first" usually yields a near-optimal solution quickly.
    
    # Let's run the solver with a generous max_scope
    max_scope = 200
    solution = solve_with_scope_limit(max_scope)
    
    if solution:
        return solution
    else:
        # If no solution found within time (unlikely for this size), 
        # or if we want to try harder, we could implement iterative deepening.
        # But for 4x8, it should be fast.
        
        # Fallback: Return the seed-like greedy result if search failed?
        # The search should not fail unless time runs out.
        # If time runs out, solve_with_scope_limit returns None.
        
        # Let's try a simpler greedy fallback if the complex search times out
        # But the complex search is the main one.
        
        # If we are here, we timed out.
        # We should return the best partial result?
        # The task says "return the best result found".
        # My current code returns None if it times out.
        # I need to track the best valid complete solution found so far.
        
        pass

    # Revised to track best solution during search
    
    best_solution = None
    
    def solve_with_scope_limit(max_scope):
        nonlocal best_solution, deadline
        
        rows = []
        
        def backtrack(row_idx, current_row, row_diffs, global_diffs):
            nonlocal best_solution
            
            if time.time() > deadline:
                return None
                
            if row_idx == n_rows:
                # Found a valid solution
                # Check if it's better than current best
                current_scope = max(max(r) for r in rows)
                if best_solution is None or current_scope < max(max(r) for r in best_solution):
                    best_solution = [r[:] for r in rows]
                # We can stop here if we want the first one, 
                # but to find the MINIMAL, we should continue?
                # No, "smallest first" means the first one found is likely minimal or close.
                # To be safe, we return it.
                return [r[:] for r in rows]
                
            if not current_row:
                return backtrack(row_idx, [0], set(), global_diffs)
                
            if len(current_row) == k + 1:
                if row_diffs & global_diffs:
                    return None
                    
                rows.append(current_row)
                new_global = global_diffs.union(row_diffs)
                
                result = backtrack(row_idx + 1, [], set(), new_global)
                if result:
                    return result
                    
                rows.pop()
                return None
                
            last_val = current_row[-1]
            for x in range(last_val + 1, max_scope + 1):
                new_diffs = set()
                valid = True
                for y in current_row:
                    d = x - y
                    if d in row_diffs or d in global_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                    
                if valid:
                    result = backtrack(row_idx, current_row + [x], row_diffs.union(new_diffs), global_diffs)
                    if result:
                        return result
            
            return None

        return backtrack(0, [], set(), set())

    solution = solve_with_scope_limit(200)
    
    if solution:
        return solution
    
    # If we timed out and found no complete solution, 
    # we might have a partial one? No, we only record complete ones.
    # If we found a complete one, we returned it.
    # If we didn't, we return None? 
    # The task requires a valid list.
    # We should ensure we always return something valid.
    
    # Fallback to a simple greedy if backtracking fails (shouldn't happen)
    used, rows = set(), []
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

# To ensure we catch the "best" if we keep searching, we can modify the logic
# to not stop at the first solution, but the first solution with "smallest first"
# is typically the best one. Stopping early saves time.

# Let's refine the fallback and ensure the function signature is correct.

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_solution = None
    best_scope = float('inf')
    
    def try_solve(max_scope):
        nonlocal best_solution, best_scope, deadline
        
        rows = []
        
        def backtrack(row_idx, current_row, row_diffs, global_diffs):
            nonlocal best_solution, best_scope
            
            if time.time() > deadline:
                return False
                
            if row_idx == n_rows:
                current_scope = max(max(r) for r in rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_solution = [r[:] for r in rows]
                # Return True to indicate a solution was found
                # We could continue searching for better ones, but 
                # given the "smallest first" heuristic, this is likely optimal.
                return True
                
            if not current_row:
                return backtrack(row_idx, [0], set(), global_diffs)
                
            if len(current_row) == k + 1:
                if row_diffs & global_diffs:
                    return False
                    
                rows.append(current_row)
                new_global = global_diffs.union(row_diffs)
                
                result = backtrack(row_idx + 1, [], set(), new_global)
                if result:
                    return True
                    
                rows.pop()
                return False
                
            last_val = current_row[-1]
            # Pruning: if last_val + (k + 1 - len(current_row)) > max_scope, we can't fit?
            # Not strictly, as gaps can be 1.
            for x in range(last_val + 1, max_scope + 1):
                new_diffs = set()
                valid = True
                for y in current_row:
                    d = x - y
                    if d in row_diffs or d in global_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                    
                if valid:
                    result = backtrack(row_idx, current_row + [x], row_diffs.union(new_diffs), global_diffs)
                    if result:
                        return True
            
            return False

        return backtrack(0, [], set(), set())

    # Try to find a solution
    # We start with a high limit to ensure we find *a* solution
    # The search order (smallest x first) will naturally find small scopes.
    found = try_solve(200)
    
    if best_solution:
        return best_solution
    
    # Fallback
    used, rows = set(), []
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
