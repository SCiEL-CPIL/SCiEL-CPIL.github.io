import time
import sys
import random

# Increase recursion limit for deep backtracking if necessary
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill
    
    num_elements = k + 1
    
    # Lower bound for scope
    # Counting bound: We need n_rows * (k+1 choose 2) distinct differences.
    # Max difference is M. So M(M-1)/2 >= n_rows * (k+1 choose 2).
    # For n=7, k=6: 7 * 21 = 147 diffs. M(M-1)/2 >= 147 -> M(M-1) >= 294 -> M~18? 
    # Wait, the counting bound in the prompt is 105. 
    # The prompt says "simple counting bound is 105". 
    # Let's trust the prompt's context or just start from a reasonable guess.
    # Actually, for Golomb rulers, the length grows quadratically.
    # For DTS, the scope is larger.
    # Let's start from 105 as suggested by the prompt's "simple counting bound".
    
    # We will iterate scope M from lower_bound upwards.
    # If we find a solution, return it.
    
    def solve(scope_limit):
        """
        Try to find a solution with max entry <= scope_limit.
        Returns list of rows or None.
        """
        used_diffs = set()
        rows = []
        
        # We will build rows one by one.
        # To break symmetry, we can enforce that the last element of each row is non-decreasing.
        # last_vals[i] will be the last element of row i.
        
        def backtrack(row_idx, last_val_constraint):
            # If we have built all rows, we are done
            if row_idx == n_rows:
                return True
            
            # Try to build a row that ends with a value >= last_val_constraint
            # and <= scope_limit
            # To optimize, we try smaller end values first to keep the scope tight?
            # Actually, we are fixing the scope_limit. We just need ANY valid row.
            # However, to help subsequent rows, maybe we should try to use "dense" rows or "sparse" rows?
            # Let's just iterate possible last values? No, that's too many.
            
            # Better: Generate a valid row starting with 0, ending with some value <= scope_limit.
            # We use a recursive helper to build the row.
            
            # To break symmetry further: 
            # The set of differences used by previous rows is fixed.
            # We need to pick a set of k+1 numbers {0, x1, ..., xk} such that:
            # 1. 0 < x1 < ... < xk <= scope_limit
            # 2. xk >= last_val_constraint (symmetry breaking)
            # 3. All differences are distinct and not in used_diffs.
            
            # We will iterate xk from last_val_constraint to scope_limit?
            # This might be slow if the range is large.
            
            # Alternative: Just build the row greedily/backtracking without fixing xk first.
            # But we need to ensure xk >= last_val_constraint.
            
            def build_row(pos, current_row, current_max):
                if pos == num_elements:
                    # Row is complete
                    last_val = current_row[-1]
                    if last_val < last_val_constraint:
                        return False
                    
                    # Check differences
                    new_diffs = set()
                    for i in range(len(current_row)):
                        for j in range(i + 1, len(current_row)):
                            d = current_row[j] - current_row[i]
                            if d in used_diffs:
                                return False
                            new_diffs.add(d)
                    
                    # Valid row found
                    rows.append(current_row)
                    used_diffs.update(new_diffs)
                    
                    if backtrack(row_idx + 1, last_val):
                        return True
                    
                    # Backtrack
                    rows.pop()
                    used_diffs.difference_update(new_diffs)
                    return False
                
                # Determine the range for the next element
                min_val = current_row[-1] + 1 if current_row else 1
                # Max possible value for this position to still fit in scope_limit?
                # We need to place (num_elements - pos - 1) more elements after this.
                # So max_val = scope_limit - (num_elements - pos - 1)
                max_val = scope_limit - (num_elements - pos - 1)
                
                # If min_val > max_val, no solution
                if min_val > max_val:
                    return False
                
                # Try values in increasing order (to find smaller scopes faster? 
                # But we are fixed on scope_limit. 
                # However, choosing smaller values might leave more room for later rows?
                # Or choosing larger values might pack differences better?
                # Let's try increasing order first.
                
                # Optimization: Skip values that create a difference already in used_diffs
                # We can check this on the fly.
                
                for val in range(min_val, max_val + 1):
                    # Check if val - x for all x in current_row are not in used_diffs
                    valid = True
                    for x in current_row:
                        d = val - x
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    if build_row(pos + 1, current_row + [val], val):
                        return True
                
                return False
            
            return build_row(1, [0], 0)
        
        return backtrack(0, 0)

    # Iterate scope from lower bound to upper bound
    # The prompt says best known is 112.
    # Let's start from 105.
    # If we run out of time, return the best valid solution found (even if not optimal).
    
    best_solution = None
    
    # We can also try a greedy heuristic first to get a valid solution quickly,
    # so we have a fallback.
    def greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Get a baseline
    baseline = greedy_solution()
    best_scope = max(max(r) for r in baseline)
    
    # Try to improve with backtracking
    # Start from a lower bound. 
    # The counting bound for M is roughly sqrt(2 * n_rows * k * (k+1) / 2) ?
    # Total diffs = n_rows * k * (k+1) / 2.
    # Max diff is M. Number of possible diffs is M(M-1)/2.
    # So M(M-1)/2 >= n_rows * k * (k+1) / 2.
    # M(M-1) >= n_rows * k * (k+1).
    # For n=7, k=6: 7 * 6 * 7 = 294.
    # M ~ sqrt(294) ~ 17. This is a very loose bound because diffs must be distinct globally, 
    # but they are drawn from a specific structure. The prompt says 105.
    
    # Let's start from 105.
    for M in range(105, 150):
        if time.time() > DEADLINE:
            break
        
        # Check time periodically
        if time.time() > DEADLINE - 5:
            break
            
        # Attempt to solve with scope M
        # We need to reset the state for each M
        # The solve function above modifies `rows` and `used_diffs` which are in the closure.
        # We need to make them local or reset them.
        
        # Let's refactor solve to be self-contained or reset state.
        # I'll create a nested function that takes M and returns the solution or None.
        
        def attempt_solve(M_limit):
            used_diffs_local = set()
            rows_local = []
            
            def backtrack_local(row_idx, last_val_constraint):
                if row_idx == n_rows:
                    return True
                
                def build_row_local(pos, current_row):
                    if pos == num_elements:
                        last_val = current_row[-1]
                        if last_val < last_val_constraint:
                            return False
                        
                        new_diffs = set()
                        for i in range(len(current_row)):
                            for j in range(i + 1, len(current_row)):
                                d = current_row[j] - current_row[i]
                                if d in used_diffs_local:
                                    return False
                                new_diffs.add(d)
                        
                        rows_local.append(current_row)
                        used_diffs_local.update(new_diffs)
                        
                        if backtrack_local(row_idx + 1, last_val):
                            return True
                        
                        rows_local.pop()
                        used_diffs_local.difference_update(new_diffs)
                        return False
                    
                    min_val = current_row[-1] + 1 if current_row else 1
                    max_val = M_limit - (num_elements - pos - 1)
                    
                    if min_val > max_val:
                        return False
                    
                    # Optimization: If we have time, we can randomize the order of values 
                    # to escape local minima? No, backtracking is exhaustive.
                    # But we can order candidates by "promisingness".
                    # Smaller values are generally better for keeping scope low, but we are fixed on M.
                    # However, using smaller values might leave larger gaps for later rows?
                    
                    for val in range(min_val, max_val + 1):
                        valid = True
                        for x in current_row:
                            d = val - x
                            if d in used_diffs_local:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        if build_row_local(pos + 1, current_row + [val]):
                            return True
                    
                    return False
                
                return build_row_local(1, [0])
            
            if backtrack_local(0, 0):
                return rows_local
            return None
        
        sol = attempt_solve(M)
        if sol is not None:
            best_solution = sol
            best_scope = M
            break # Found the smallest M, so we are done
    
    if best_solution is None:
        return baseline
    
    return best_solution
