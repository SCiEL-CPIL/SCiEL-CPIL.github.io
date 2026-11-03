"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
import time
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct across
    all rows. Minimize the largest entry.
    
    Uses Iterative Deepening Backtracking with Symmetry Breaking.
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on max entry
    # We need total_diffs distinct values in 1..M
    # So M must be at least total_diffs
    # Wait, the differences are from 1 to M.
    # So we need M >= total_diffs.
    # 7 * 6 * 7 / 2 = 147.
    # So M >= 147? No.
    # The differences are $a_j - a_i$. The max difference is $M$.
    # The set of all differences across all rows must be distinct.
    # The union of differences is a subset of $\{1, 2, ..., M\}$.
    # So the size of the union is at most $M$.
    # We need $total\_diffs$ distinct differences.
    # So $M \ge total\_diffs$.
    # 147 differences. So M >= 147.
    # But the problem says "best known set has largest entry 112".
    # This implies my lower bound logic is wrong or the problem allows something else?
    # "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows"
    # Yes, distinct over all rows.
    # If M=112, the possible differences are 1..112.
    # We need 147 distinct differences.
    # This is impossible if differences must be in 1..112.
    # Let me re-read carefully.
    # "all positive differences ... are distinct over all rows"
    # If there are 147 differences, and they are all distinct integers, they must be 1..147 at minimum?
    # No, they don't have to be 1..147. They just have to be distinct.
    # But they are bounded by M.
    # So we need a set of 147 distinct integers, all $\le M$.
    # The smallest possible max value for 147 distinct positive integers is 147 (1, 2, ..., 147).
    # So M must be at least 147.
    # Why does the prompt say "best known set has largest entry 112"?
    # And "simple counting bound is 105"?
    # 105 < 112 < 147.
    # There is a contradiction in my understanding or the prompt's numbers.
    
    # Let's check the counting bound logic in the prompt.
    # "simple counting bound is 105".
    # Maybe the differences don't have to be distinct *globally*?
    # "all positive differences ... are distinct over all rows"
    # This usually means globally distinct.
    
    # Let's look at the example or standard definition of Difference Triangle Sets.
    # A Difference Triangle Set (DTS) is a set of sets.
    # Often defined as: A set of $n$ subsets of $\{0, ..., M\}$ of size $k+1$ such that all differences are distinct.
    # If so, M >= 147 is a hard lower bound.
    
    # Is it possible the prompt implies something else?
    # "Difference triangle set (7,5)" -> n=7, k=5?
    # The prompt says: "Choose 7 rows of 6 increasing integers".
    # So k=5 (indices 0..5)? No, "6 increasing integers" means size 6.
    # So $k+1=6 \implies k=5$.
    # Let's re-read the interface: `construct(n_rows, k)`.
    # "Each row is a list of k + 1 integers".
    # So if the task is (7,5), then n_rows=7, k=5.
    # Size of row = 6.
    # Number of differences per row = $5 \times 6 / 2 = 15$.
    # Total differences = $7 \times 15 = 105$.
    # Ah! The prompt says "Difference triangle set (7,5)".
    # And "Choose 7 rows of 6 increasing integers".
    # So $n=7, k=5$.
    # Total diffs = 105.
    # Lower bound M >= 105.
    # Best known 112.
    # This makes sense.
    
    # So in the code, `k` passed to `construct` is 5.
    # The prompt says: "The evaluator passes this task's n_rows and k."
    # So I should use the passed `k`.
    
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs
    
    # Upper bound heuristic
    upper_bound = lower_bound + 20
    
    import time
    start_time = time.time()
    time_limit = 110
    
    # We will try M from lower_bound to upper_bound
    for M in range(lower_bound, upper_bound + 1):
        if time.time() - start_time > time_limit:
            break
            
        # State for backtracking
        # used_diffs: bitmask or set. Since M <= ~120, we can use a set or an integer bitmask.
        # Python integers are arbitrary precision, so bitmask is efficient.
        used_diffs = 0
        rows = []
        prev_row = None
        
        # DFS to find a valid configuration for this M
        # We use symmetry breaking: rows must be lexicographically increasing.
        
        def dfs_row_idx(row_idx, used_diffs, prev_row, rows):
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            # Generate possible rows for this index
            # The row must be strictly increasing, start at 0, end <= M.
            # And lexicographically > prev_row.
            
            # We backtrack on the elements of the current row.
            # current_row: list of elements chosen so far.
            # col_idx: current position in the row (0 to k).
            
            def dfs_row_col(col_idx, current_row, used_diffs_local, prev_row, rows):
                if time.time() - start_time > time_limit:
                    return False
                
                if col_idx == k + 1:
                    # Check symmetry breaking
                    if prev_row is not None:
                        if current_row <= prev_row:
                            return False
                    
                    # Check if all differences are distinct (handled during construction)
                    # Recurse to next row
                    rows.append(current_row)
                    if dfs_row_idx(row_idx + 1, used_diffs_local, current_row, rows):
                        return True
                    rows.pop()
                    return False
                
                # Determine start value
                if col_idx == 0:
                    start_val = 0
                    end_val = 0
                else:
                    start_val = current_row[-1] + 1
                    end_val = M
                
                # Optimization: If we are at the last column, we just pick the smallest valid?
                # No, we need to try all to find a solution.
                
                # Pruning:
                # The remaining columns need to fit in M.
                # If current_row[-1] + (k - col_idx) > M, it's impossible.
                if col_idx > 0 and current_row[-1] + (k - col_idx) > M:
                    return False
                
                for x in range(start_val, end_val + 1):
                    # Check differences with previous elements in current_row
                    valid = True
                    new_diffs_mask = 0
                    for y in current_row:
                        d = x - y
                        # d must be > 0
                        # Check if d is in used_diffs_local
                        if (used_diffs_local >> d) & 1:
                            valid = False
                            break
                        new_diffs_mask |= (1 << d)
                    
                    if not valid:
                        continue
                    
                    # Additional pruning: 
                    # Can we complete the row?
                    # The next elements must be > x.
                    # The max possible value is M.
                    # We have k - col_idx elements left.
                    # The smallest they can be is x+1, x+2, ...
                    # So we need x + (k - col_idx) <= M.
                    if x + (k - col_idx) > M:
                        break # Since x is increasing, further x will also fail
                
                    # Recurse
                    if dfs_row_col(col_idx + 1, current_row + [x], used_diffs_local | new_diffs_mask, prev_row, rows):
                        return True
            
            # Start row construction
            # Symmetry breaking for the first row?
            # If prev_row is None, we can fix some elements to reduce symmetry?
            # For the first row, we can assume the first element is 0 (already done).
            # We can also assume the second element is 1? No, not necessarily.
            # But we can fix the row to be "minimal" in some sense?
            # Lexicographic ordering of rows is sufficient.
            
            return dfs_row_col(0, [0], used_diffs, prev_row, rows)
        
        if dfs_row_idx(0, 0, None, rows):
            return rows
    
    # Fallback if no solution found in time or range
    # Use the greedy approach from the original program as a safety net
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

# EVOLVE-BLOCK-END
