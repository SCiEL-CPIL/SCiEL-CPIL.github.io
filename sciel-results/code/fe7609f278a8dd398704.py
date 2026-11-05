import time
import sys

def construct(n_rows, k):
    """
    Finds a difference triangle set (n_rows, k) by minimizing the largest entry.
    Uses backtracking with symmetry breaking and pruning.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # Lower bound for M: we need n_rows * C(k+1, 2) distinct differences
    # The smallest possible differences are 1, 2, ..., N_diff
    # So M must be at least N_diff.
    # Also, a single row of k+1 marks has k differences, max diff is M.
    # The smallest possible M for a single row is k (marks 0,1,...,k).
    # So M >= max(k, n_rows * C(k+1, 2))
    
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # A better lower bound: the sum of the smallest N_diffs differences is not directly relevant,
    # but the largest difference must be at least the N_diffs-th smallest difference.
    # If we use differences 1, 2, ..., N_diffs, the largest is N_diffs.
    # So M >= total_diffs is a valid lower bound.
    
    M_lower = max(k, total_diffs)
    
    best_rows = None
    best_M = float('inf')
    
    # We will try M values from M_lower upwards
    # To be safe, we can cap the search if it takes too long, but we hope to find it quickly.
    # The known best for (4,7) is 134. total_diffs = 4*28 = 112.
    
    # We'll use a backtracking function to find n_rows rows for a given M
    def find_rows_for_M(M):
        # used_diffs is a set of differences already used
        used_diffs = set()
        rows = []
        
        # Symmetry breaking: we require rows to be in lexicographical order
        # So row i must be >= row i-1 lexicographically
        
        def backtrack(row_idx, current_row, last_row):
            if row_idx == n_rows:
                return True
            
            # If this is not the first row, current_row must be >= last_row
            # We enforce this by only allowing marks that keep the row >= last_row
            # A simple way: when choosing the next mark, ensure the partial row is >= last_row
            # Actually, it's easier to generate valid rows and check at the end, but that's slow.
            # Better: when extending current_row, ensure that if len(current_row) == len(last_row), 
            # then current_row >= last_row. But since all rows have the same length, we can just 
            # ensure that the final row is >= last_row.
            # To prune early, we can check if the current_row (padded with 0s or something) is >= last_row.
            # Simpler: just check at the end of the row.
            
            # Generate possible next marks for current_row
            # current_row starts with [0]
            # The next mark x must be > current_row[-1]
            # And the differences {x - a for a in current_row} must not be in used_diffs
            
            # Heuristic: try smaller x first? Or larger? 
            # Smaller x uses smaller differences, which might be more scarce.
            # Let's try smaller x first.
            
            # To break symmetry further, we can restrict the first mark of the row.
            # But the row always starts with 0.
            
            # We need to choose k more marks.
            # Let's use a recursive function to fill the row.
            
            def fill_row(pos, row_so_far):
                if pos == k + 1:
                    # Check symmetry breaking: row_so_far >= last_row (if last_row is not None)
                    if last_row is not None and row_so_far < last_row:
                        return False
                    
                    # Check if all differences are new
                    row_diffs = set()
                    for i in range(len(row_so_far)):
                        for j in range(i+1, len(row_so_far)):
                            d = row_so_far[j] - row_so_far[i]
                            if d in used_diffs:
                                return False
                            row_diffs.add(d)
                    
                    # If we get here, the row is valid
                    used_diffs.update(row_diffs)
                    rows.append(row_so_far)
                    
                    if backtrack(row_idx + 1, [0], row_so_far):
                        return True
                    
                    # Undo
                    rows.pop()
                    used_diffs.difference_update(row_diffs)
                    return False
                
                # Try possible next marks
                # The next mark must be > row_so_far[-1]
                # And <= M
                # To speed up, we can estimate the minimum possible next mark.
                # The remaining positions are k+1 - pos.
                # The smallest possible next mark is row_so_far[-1] + 1.
                # The largest possible next mark is M.
                
                # Pruning: if the remaining differences cannot fit in the remaining slots, prune.
                # This is complex. Let's just iterate.
                
                # To avoid symmetry, we can also enforce that the first few marks are in a certain range.
                # But let's keep it simple.
                
                start_x = row_so_far[-1] + 1
                end_x = M + 1
                
                # Optimization: if we have already tried a row with a larger first mark, 
                # we can skip smaller ones? No, rows are independent except for differences.
                
                for x in range(start_x, end_x):
                    # Check if differences with x are available
                    new_diffs = {x - a for a in row_so_far}
                    if new_diffs & used_diffs:
                        continue
                    
                    # Also, we can prune if x is too small to allow the remaining marks to be distinct and <= M.
                    # The remaining marks must be > x, and <= M.
                    # So we need at least (k+1 - pos - 1) marks in (x, M].
                    # So M - x >= (k+1 - pos - 1)
                    if M - x < (k + 1 - pos - 1):
                        break # No need to try larger x? No, larger x leaves less room.
                        # Actually, if M - x is too small, then even larger x will be worse.
                        # So we can break.
                    
                    # Recurse
                    if fill_row(pos + 1, row_so_far + [x]):
                        return True
                
                return False
            
            # Start filling the row
            # We can also use a heuristic to choose the order of rows.
            # But let's just fill row_idx.
            
            # To speed up, we can restrict the first mark of the row (which is 0) and the second mark.
            # The second mark must be >= 1.
            # Also, for symmetry breaking, the row must be >= last_row.
            # So the second mark must be >= last_row[1] if last_row exists.
            
            # Let's just call fill_row
            if fill_row(1, [0]):
                return True
            
            return False
        
        return backtrack(0, [0], None)
    
    # Try M values
    # We can start from M_lower and go up.
    # To avoid infinite loop, we can cap the number of M values tried.
    # But the problem is small, so it should be fine.
    
    for M in range(M_lower, 200):  # Cap at 200 for safety
        if time.time() > deadline:
            break
        
        # Clear state
        used_diffs = set()
        rows = []
        
        # We need to reset the state for each M
        # The backtrack function uses closures, so we need to re-define or reset.
        # Let's refactor to avoid closure issues.
        
        # For simplicity, let's just call a new function for each M.
        # But the above code has closures. Let's move the logic to a helper.
        
        # Actually, the code above is a bit messy. Let's rewrite it cleanly.
        
        pass  # Placeholder, we'll implement the clean version below
    
    # Given the complexity and time limit, a simpler approach might be better:
    # Use a greedy approach with multiple random restarts and local search.
    # But the task asks for a structural change.
    
    # Let's stick with the backtracking but implement it cleanly.
    
    # For the sake of this response, I'll provide a clean backtracking implementation.
    
    # If the backtracking is too slow, we can fall back to the greedy method.
    
    # Let's implement the clean backtracking.
    
    def solve_for_M(M):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row, last_row):
            if row_idx == n_rows:
                return True
            
            # Fill the current row
            def fill_row(pos, row_so_far):
                if pos == k + 1:
                    # Check symmetry breaking
                    if last_row is not None and row_so_far < last_row:
                        return False
                    
                    # Check differences
                    row_diffs = set()
                    for i in range(len(row_so_far)):
                        for j in range(i+1, len(row_so_far)):
                            d = row_so_far[j] - row_so_far[i]
                            if d in used_diffs:
                                return False
                            row_diffs.add(d)
                    
                    used_diffs.update(row_diffs)
                    rows.append(row_so_far)
                    
                    if backtrack(row_idx + 1, [0], row_so_far):
                        return True
                    
                    rows.pop()
                    used_diffs.difference_update(row_diffs)
                    return False
                
                start_x = row_so_far[-1] + 1
                # Pruning: if M - x < remaining_marks, break
                remaining_marks = k + 1 - pos - 1
                for x in range(start_x, M + 1):
                    if M - x < remaining_marks:
                        break
                    
                    new_diffs = {x - a for a in row_so_far}
                    if new_diffs & used_diffs:
                        continue
                    
                    if fill_row(pos + 1, row_so_far + [x]):
                        return True
                
                return False
            
            return fill_row(1, [0])
        
        return backtrack(0, [0], None)
    
    for M in range(M_lower, 200):
        if time.time() > deadline:
            break
        
        if solve_for_M(M):
            best_rows = rows[:]  # Copy the rows
            best_M = M
            break
    
    if best_rows is None:
        # Fallback to greedy if backtracking fails or times out
        # This ensures we always return a valid solution
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
        best_rows = rows
    
    return best_rows
