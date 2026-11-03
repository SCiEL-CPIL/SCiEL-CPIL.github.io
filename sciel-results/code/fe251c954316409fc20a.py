"""Backtracking solver for Difference Triangle Sets."""
import sys
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all pairwise differences
    within rows are distinct across all rows.
    
    Uses Depth-First Search with backtracking and pruning.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * n_rows] if k == 0 else []
    
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    counting_bound = total_diffs
    
    # We will try scopes starting from counting_bound upwards.
    # The known best for (7,5) is 112. Counting bound is 7 * 15 = 105.
    # We'll try up to counting_bound + 50 to be safe, or until found.
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    def _solve_scope(scope):
        """
        Attempt to construct n_rows rows with max value <= scope.
        Returns list of rows if successful, None otherwise.
        """
        used_diffs = set()
        rows = []
        
        def dfs(row_idx, current_row, current_diffs):
            """
            row_idx: index of the row being constructed
            current_row: list of elements in the current row so far
            current_diffs: set of differences in the current row so far
            """
            if time.time() - start_time > time_limit:
                return False
                
            if row_idx == n_rows:
                return True
                
            if len(current_row) == k + 1:
                # Row complete, move to next row
                rows.append(current_row)
                # Add diffs to global used set
                for d in current_diffs:
                    used_diffs.add(d)
                
                if dfs(row_idx + 1, [0], set()):
                    return True
                
                # Backtrack
                rows.pop()
                for d in current_diffs:
                    used_diffs.discard(d)
                return False
                
            # Determine next element
            last_val = current_row[-1]
            # We need to pick a value v such that last_val < v <= scope
            # Also, v - u for all u in current_row must be new
            
            # Heuristic: Try values from last_val + 1 to scope
            for v in range(last_val + 1, scope + 1):
                new_diffs = set()
                valid = True
                for u in current_row:
                    d = v - u
                    if d in used_diffs or d in current_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                    
                # Check if we can still complete the row
                # Remaining elements needed: k + 1 - (len(current_row) + 1)
                remaining_needed = k + 1 - (len(current_row) + 1)
                # Available numbers: scope - v
                if scope - v < remaining_needed:
                    continue
                    
                # Recursive step
                # Pass a new set of diffs
                if dfs(row_idx, current_row + [v], current_diffs.union(new_diffs)):
                    return True
                    
            return False

        # Start with first row
        if dfs(0, [0], set()):
            return rows
        return None

    for scope in range(counting_bound, counting_bound + 100):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a solution with this scope
        result = _solve_scope(scope)
        if result is not None:
            return result
            
    # If no solution found within time or scope limit, fall back to a heuristic construction
    # This should rarely happen for small n_rows, k
    return _heuristic_fallback(n_rows, k)

def _heuristic_fallback(n_rows, k):
    """
    Simple greedy construction if backtracking fails or times out.
    """
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
