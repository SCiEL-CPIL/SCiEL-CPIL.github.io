import time
import random
import math

def construct(n_rows, k):
    deadline = time.time() + 115
    best_rows = None
    best_scope = float('inf')
    
    def _scope(rows):
        return max(max(row) for row in rows)
    
    def _compute_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def _is_valid(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    def _greedy_construct(n_rows, k, seed=0):
        rng = random.Random(seed)
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
    
    def _csp_backtrack(n_rows, k, deadline, max_scope=None):
        """
        Backtracking search with forward checking.
        We try to find a valid assignment where the largest entry is <= max_scope.
        Variables: (row_idx, pos_idx) for pos_idx in 1..k
        Order: Row by row, position by position.
        """
        if max_scope is None:
            # Estimate a loose upper bound
            max_scope = 150 * n_rows * k
            
        # Pre-compute nothing, just do it
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        
        # We fill row 0, then row 1, etc.
        # To help with pruning, we might want to order rows by difficulty?
        # But rows are symmetric, so order doesn't matter for validity, only for search space.
        # However, the "largest entry" constraint is global.
        
        # To minimize the largest entry, we should ideally constrain the maximum value early.
        # But here we are checking if a solution exists within max_scope.
        # We will search for any solution.
        
        # Heuristic: Fill the row with the most constraints first? 
        # Actually, since rows are independent except for the diff set, 
        # the order of rows doesn't affect the validity of the final set, 
        # but it affects the search path.
        
        # Let's try filling row 0 completely, then row 1, etc.
        # This is a standard DFS.
        
        # Optimization: If we reach a state where the current row's last element + (k - pos) > max_scope, prune?
        # No, we can always pick smaller values? No, values must increase.
        # So if row[-1] + (k - pos) > max_scope, we can't fit the remaining k-pos elements in increasing order within max_scope.
        
        def backtrack(row_idx, pos_idx):
            # Check time
            if time.time() > deadline - 2:
                return False
                
            # If we have filled all rows
            if row_idx == n_rows:
                return True
            
            row = rows[row_idx]
            
            # If we are at the start of a new row
            if pos_idx == 0:
                # We need to fill positions 1..k for this row
                # Start with row = [0]
                # Try to fill pos 1
                return backtrack(row_idx, 1)
            
            # If we are filling position pos_idx (1..k)
            # The previous value is row[pos_idx - 1]
            prev_val = row[pos_idx - 1] if pos_idx > 0 else 0
            
            # Minimum possible value
            min_val = prev_val + 1
            
            # Maximum possible value
            # Must be <= max_scope
            # Also, must leave room for remaining (k - pos_idx) elements
            # min_val + (k - pos_idx) <= max_scope
            max_val = min(max_scope, max_scope - (k - pos_idx))
            
            # If min_val > max_val, no solution
            if min_val > max_val:
                return False
            
            # Forward checking: 
            # For each candidate v in [min_val, max_val]:
            #   Check if differences v - row[i] for i in 0..pos_idx-1 are in used_diffs
            #   Also check if they are distinct within the row (implied by strict increase)
            
            # To speed up, we can iterate v and check.
            # We can also prune if v - row[0] is in used_diffs (since row[0]=0, this is just v in used_diffs)
            
            # Let's try values in increasing order to find a small scope solution if possible?
            # Or random order?
            # Increasing order is good for minimizing scope locally.
            
            for v in range(min_val, max_val + 1):
                # Check differences
                valid = True
                new_diffs = []
                for i in range(pos_idx):
                    d = v - row[i]
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # If valid, recurse
                row.append(v)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if pos_idx == k:
                    # Row complete
                    if backtrack(row_idx + 1, 0):
                        return True
                    # Backtrack
                    used_diffs.difference_update(new_diffs)
                    row.pop()
                else:
                    if backtrack(row_idx, pos_idx + 1):
                        return True
                    used_diffs.difference_update(new_diffs)
                    row.pop()
            
            return False
        
        # Start search
        # Initialize rows
        rows = [[0] for _ in range(n_rows)]
        
        # We need to be careful with the recursion depth and Python's recursion limit
        # n_rows * k is small (7 * 5 = 35), so depth is fine.
        
        # However, the branching factor can be large.
        # We should add a node limit or rely on time.
        
        # Let's wrap it in a try/except just in case of recursion depth issues (unlikely here)
        try:
            if backtrack(0, 0):
                # Extract rows
                result = []
                for r in rows:
                    # r has length k+1
                    result.append(r)
                return result
        except RecursionError:
            pass
            
        return None

    def _simulated_annealing(rows, n_rows, k, deadline):
        """Simulated annealing to escape local minima."""
        best_rows = [list(r) for r in rows]
        best_scope = _scope(best_rows)
        current_rows = [list(r) for r in rows]
        current_scope = _scope(current_rows)
        
        rng = random.Random(42)
        start_temp = 50.0
        end_temp = 0.1
        cooling_rate = 0.995
        
        # Precompute diffs for current
        row_diffs = [_compute_diffs(r) for r in current_rows]
        all_diffs = set()
        for rd in row_diffs:
            all_diffs |= rd
        
        temp = start_temp
        iterations = 0
        max_iterations = 50000
        
        while time.time() < deadline - 10 and temp > end_temp and iterations < max_iterations:
            iterations += 1
            
            # Pick a random row and position to modify
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            if len(row) <= 1:
                continue
            
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            
            # Try a random delta
            delta = rng.randint(-15, 15)
            if delta == 0:
                continue
            new_val = old_val + delta
            
            # Check validity
            if new_val <= 0:
                continue
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
            
            # Compute old diffs for this row
            old_diffs = _compute_diffs(row)
            
            # Apply change
            row[pos] = new_val
            new_diffs = _compute_diffs(row)
            
            # Check for internal duplicates
            expected_diff_count = len(row) * (len(row) - 1) // 2
            if len(new_diffs) != expected_diff_count:
                row[pos] = old_val
                continue
            
            # Check conflicts with other rows
            other_diffs = all_diffs - old_diffs
            if new_diffs & other_diffs:
                row[pos] = old_val
                continue
            
            # Valid move - compute new scope
            new_scope = _scope(current_rows)
            delta_scope = new_scope - current_scope
            
            # Accept if better, or with probability exp(-delta_scope/temp) if worse
            if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temp):
                # Accept the move
                all_diffs = (all_diffs - old_diffs) | new_diffs
                row_diffs[ri] = new_diffs
                current_scope = new_scope
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                # Reject the move
                row[pos] = old_val
            
            temp *= cooling_rate
        
        return best_rows, best_scope

    def _local_polish(rows, n_rows, k, deadline):
        """Final local polish with small moves to find immediate improvements."""
        best_rows = [list(r) for r in rows]
        best_scope = _scope(best_rows)
        rng = random.Random(123)
        
        row_diffs = [_compute_diffs(r) for r in best_rows]
        all_diffs = set()
        for rd in row_diffs:
            all_diffs |= rd
        
        # Try many small moves
        for _ in range(20000):
            if time.time() >= deadline - 5:
                break
            
            ri = rng.randrange(n_rows)
            row = best_rows[ri]
            if len(row) <= 1:
                continue
            
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            
            # Try small deltas
            for delta in range(-3, 4):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                
                old_diffs = _compute_diffs(row)
                row[pos] = new_val
                new_diffs = _compute_diffs(row)
                
                expected_diff_count = len(row) * (len(row) - 1) // 2
                if len(new_diffs) != expected_diff_count:
                    row[pos] = old_val
                    continue
                
                other_diffs = all_diffs - old_diffs
                if new_diffs & other_diffs:
                    row[pos] = old_val
                    continue
                
                new_scope = _scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    all_diffs = (all_diffs - old_diffs) | new_diffs
                    row_diffs[ri] = new_diffs
                    break
                else:
                    row[pos] = old_val
        
        return best_rows, best_scope

    # Phase 1: Greedy construction to get a valid starting point
    for seed in range(20):
        if time.time() >= deadline - 20:
            break
        rows = _greedy_construct(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]

    # Phase 2: CSP Backtracking to try to find a better solution directly
    # We try to find a solution with a smaller scope than the current best.
    # We do binary search on the scope? Or just try a few lower bounds.
    
    if best_rows is not None:
        # Try to find a solution with scope < best_scope
        # We can try CSP with max_scope = best_scope - 1, then best_scope - 5, etc.
        # Or just try CSP with max_scope = best_scope - 1. If it fails, we know best_scope is tight for this search.
        
        # Let's try a few decreasing scopes
        target_scopes = [best_scope - 1, best_scope - 5, best_scope - 10, best_scope - 20]
        
        for target in target_scopes:
            if target < 10:
                continue
            if time.time() >= deadline - 15:
                break
            
            # Run CSP with this target scope
            csp_result = _csp_backtrack(n_rows, k, deadline, max_scope=target)
            if csp_result is not None:
                new_scope = _scope(csp_result)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in csp_result]
                # If we found a solution, we can stop trying smaller scopes immediately?
                # No, we might find an even smaller one. But CSP is expensive.
                # Let's stop if we find one, as it's a significant jump.
                break

    # Phase 3: Simulated Annealing to polish the best found so far
    if best_rows is not None and time.time() < deadline - 10:
        best_rows, best_scope = _simulated_annealing(best_rows, n_rows, k, deadline)

    # Phase 4: Local Polish
    if best_rows is not None and time.time() < deadline - 5:
        best_rows, best_scope = _local_polish(best_rows, n_rows, k, deadline)

    if best_rows is None:
        # Fallback: Construct a valid solution with large scope
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, k + 1):
                v = row[-1] + 1
                while True:
                    valid = True
                    for prev in row:
                        d = v - prev
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(v)
                        for prev in row[:-1]:
                            used_diffs.add(v - prev)
                        break
                    v += 1
            rows.append(row)
        return rows

    return best_rows
