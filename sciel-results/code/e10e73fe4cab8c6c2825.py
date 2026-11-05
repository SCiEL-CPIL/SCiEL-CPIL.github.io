import time
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave margin for 120s limit
    
    # Theoretical lower bound for the maximum element M
    # Total differences needed: n_rows * k * (k+1) / 2
    # Max difference possible with max element M is M.
    # So we need at least n_rows * k * (k+1) / 2 distinct differences in [1, M].
    # Thus M >= n_rows * k * (k+1) / 2
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs_needed
    
    best_solution = None
    best_max = float('inf')
    
    # We will try to find a solution with max element M, starting from lower_bound
    # We iterate M upwards. If we find a valid solution for M, we can stop (since we want min M).
    # However, if the search for a specific M is too slow, we might time out.
    # To be safe, we can run backtracking for a specific M with a time check.
    # If it times out, we try a larger M? No, larger M is easier to satisfy but we want min M.
    # Actually, if we can't find a solution for M quickly, it might not exist, or it's hard.
    # But usually, if a solution exists for M, it exists for M+1.
    # So we start from LB and go up.
    
    # Helper for backtracking
    def solve(max_limit, current_rows, used_diffs, row_idx):
        """
        Try to fill rows from row_idx to n_rows-1 with max element <= max_limit.
        Returns True if a valid completion is found, updating current_rows and used_diffs.
        """
        if time.time() - start_time > time_limit:
            return False
            
        if row_idx == n_rows:
            return True
            
        # Construct row row_idx
        # We need to choose k+1 numbers: 0 = x_0 < x_1 < ... < x_k <= max_limit
        # Such that all differences x_j - x_i are in [1, max_limit] and not in used_diffs.
        # And internally distinct.
        
        # Heuristic: Try to build the row greedily with backtracking.
        # To minimize search, we pick x_1, x_2, ... x_k.
        # x_i must be > x_{i-1}.
        # Also, to allow room for the rest of the row, we should pick small x_i.
        
        # We use a recursive helper for the row construction
        def build_row(pos, current_row, row_used_diffs):
            if time.time() - start_time > time_limit:
                return False
                
            if pos == k + 1:
                # Row complete
                # Add to global state
                current_rows[row_idx] = list(current_row)
                used_diffs.update(row_used_diffs)
                
                # Recurse to next row
                if solve(max_limit, current_rows, used_diffs, row_idx + 1):
                    return True
                
                # Backtrack
                used_diffs.difference_update(row_used_diffs)
                current_rows[row_idx] = None
                return False
            
            # Determine range for current element current_row[pos]
            # If pos == 0, it's 0.
            # Else, it must be > current_row[pos-1].
            # Also, we need to ensure that we can still place the remaining (k - pos) elements
            # such that the last one is <= max_limit.
            # Max possible value for current element to allow k-pos more elements:
            # If we pick v, the next must be at least v+1, ..., last at least v + (k-pos).
            # So v + (k - pos) <= max_limit => v <= max_limit - (k - pos).
            
            if pos == 0:
                start_val = 1
                end_val = 0 # 0 is fixed
            else:
                start_val = current_row[pos-1] + 1
                # Pruning: ensure we can fit remaining elements
                # Remaining elements to place: k - pos
                # They must be strictly increasing.
                # Smallest possible sequence: v, v+1, ..., v + (k-pos)
                # Last element: v + (k-pos) <= max_limit
                # So v <= max_limit - (k - pos)
                end_val = max_limit - (k - pos)
                
                if start_val > end_val:
                    return False
            
            # Iterate possible values for current_row[pos]
            # We try smaller values first to keep the row "tight" and leave room for others?
            # Or does keeping it tight create more conflicts?
            # Usually, greedy (smallest first) is a good heuristic for packing.
            
            for v in range(start_val, end_val + 1):
                # Check internal conflicts for this row
                # New differences: v - current_row[i] for i in 0..pos-1
                new_diffs = []
                valid = True
                for i in range(pos):
                    d = v - current_row[i]
                    if d in row_used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check global conflicts
                if any(d in used_diffs for d in new_diffs):
                    continue
                
                # Tentatively accept
                current_row.append(v)
                row_used_diffs.update(new_diffs)
                
                if build_row(pos + 1, current_row, row_used_diffs):
                    return True
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    row_used_diffs.discard(d)
            
            return False
        
        # Start building the row
        initial_row = [0]
        initial_row_diffs = set()
        if build_row(1, initial_row, initial_row_diffs):
            return True
            
        return False

    # Main loop: try increasing max_limit
    # We might not find a solution for LB, LB+1, etc. immediately due to search complexity.
    # We will try each M. If solve() returns True, we have a solution with max M.
    # We return it immediately as it's the minimum possible for that M.
    
    # To handle potential long runs for a specific M, we rely on the global time check inside solve().
    # If solve() times out, it returns False. We then try M+1?
    # This is risky. If M is hard but solvable, and M+1 is easy, we might skip M.
    # But M+1 is a valid solution (just larger max). We want the SMALLEST max.
    # So if we fail M due to timeout, we shouldn't assume M is impossible.
    # However, for this specific small instance (7x6), the search space is small enough
    # that a good heuristic should find the optimal M (112) or close to it quickly.
    # The counting bound is 105.
    
    # Let's try from lower_bound to a safe upper bound (e.g., 150)
    # If we don't find anything, we fall back to a greedy heuristic result.
    
    # Fallback greedy to ensure we always return *something*
    def greedy_fallback():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                found = False
                for x in range(row[-1] + 1, 200):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                if not found:
                    # Expand
                    for x in range(row[-1] + 1, 500):
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                if not found:
                    # Should not happen
                    row.append(row[-1] + 100)
            rows.append(row)
        return rows

    # Try systematic search
    found = False
    for M in range(lower_bound, 150):
        if time.time() - start_time > time_limit:
            break
            
        current_rows = [None] * n_rows
        used_diffs = set()
        
        # We need to pass a mutable list to solve
        # solve modifies current_rows and used_diffs
        if solve(M, current_rows, used_diffs, 0):
            # Check if the solution is valid (it should be by construction)
            # And get the max
            max_val = max(max(r) for r in current_rows)
            # It should be <= M
            best_solution = current_rows
            best_max = max_val
            found = True
            break
        else:
            # If solve timed out, it returned False.
            # We can't distinguish between "no solution" and "timeout".
            # But if we timeout, we should probably stop trying larger M in this loop
            # because larger M will likely also timeout or be slower to verify?
            # Actually, larger M makes the constraint looser, so it might be easier to find a solution.
            # But the search space is larger.
            # For now, if we timeout on M, we break and use fallback or best found so far.
            # Given the small size, timeout is unlikely unless M is very close to the limit.
            pass

    if not found:
        # Use fallback
        best_solution = greedy_fallback()
        best_max = max(max(r) for r in best_solution)

    return best_solution
