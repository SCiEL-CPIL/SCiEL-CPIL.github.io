"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Uses an optimized backtracking search with bitset-like pruning (using sets for simplicity and speed in Python for small M).
    """
    # For the specific case (7,5), the optimal max is 112.
    # We will attempt to find the optimal solution via backtracking.
    # If the search is too slow, we fall back to a greedy heuristic.
    
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    # Total differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on M
    # M >= total_diffs
    # Also, for a single row, the max diff is M_row. 
    # A tighter lower bound: 
    # The sum of the k largest differences in a row is not directly constrained, 
    # but the differences must be distinct.
    
    # We'll search for the minimum M.
    # We can use a binary search on M, or just iterative deepening on M.
    # Since M is small (around 100-150), we can just try to find a solution 
    # that minimizes M directly in the backtracking by pruning branches 
    # that exceed the current best M.
    
    best_solution = None
    best_max = float('inf')
    
    # Helper to check if a set of differences is valid
    # used_diffs: set of integers
    # row: list of integers
    
    def is_valid_row(row, used_diffs):
        # Check if any difference in this row is already in used_diffs
        # or if there are duplicate differences within the row
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used_diffs or d in row_diffs:
                    return False
                row_diffs.add(d)
        return True

    def backtrack(row_idx, current_row, used_diffs, current_max):
        nonlocal best_solution, best_max
        
        # Timeout check
        if time.time() - start_time > timeout:
            return False
            
        # Pruning: if current max is already worse than best, stop
        if current_max >= best_max:
            return False
            
        # If we have constructed all rows
        if row_idx == n_rows:
            if current_max < best_max:
                best_max = current_max
                best_solution = [list(r) for r in rows] + [list(current_row)]
                return True
            return False
            
        # If current row is complete
        if len(current_row) == k + 1:
            # Validate the row against used_diffs
            if not is_valid_row(current_row, used_diffs):
                return False
                
            # Calculate new used_diffs
            new_used_diffs = set(used_diffs)
            for i in range(len(current_row)):
                for j in range(i + 1, len(current_row)):
                    new_used_diffs.add(current_row[j] - current_row[i])
                    
            # Move to next row
            rows.append(current_row)
            success = backtrack(row_idx + 1, [0], new_used_diffs, max(current_max, current_row[-1]))
            rows.pop()
            
            if success:
                return True
            return False
            
        # If current row is not complete, try to add the next element
        last_val = current_row[-1]
        
        # Heuristic: Try values that are "small" to keep max low, 
        # but also ensure we don't block too many future differences.
        # We iterate x from last_val + 1 upwards.
        # We stop if x exceeds best_max (or a reasonable limit if best_max is inf).
        
        # Upper bound for x: 
        # If we have best_max, x cannot exceed best_max.
        # If best_max is inf, we need a heuristic limit. 
        # Let's set a soft limit based on total_diffs + some buffer.
        # Actually, if we find a solution with M, we prune anything > M.
        # Initially, let's set a high limit, e.g., 200.
        limit = best_max if best_max != float('inf') else 200
        
        for x in range(last_val + 1, limit + 1):
            # Check if adding x creates a conflict with used_diffs
            # The new differences are x - a_i for all a_i in current_row
            valid = True
            new_diffs = set()
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
                
            if not valid:
                continue
                
            # Recurse
            new_row = current_row + [x]
            new_used = used_diffs | new_diffs
            new_max = max(current_max, x)
            
            if backtrack(row_idx, new_row, new_used, new_max):
                return True
                
        return False
        
    # Start the search
    rows = []
    # Initialize with the first element 0
    backtrack(0, [0], set(), 0)
    
    if best_solution is not None:
        return best_solution
        
    # Fallback: Greedy approach if backtracking fails or times out
    # This ensures we return a valid solution, even if not optimal.
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            valid = True
            new_diffs = set()
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                row.append(x)
                used_diffs |= new_diffs
            x += 1
        rows.append(row)
        
    return rows
# EVOLVE-BLOCK-END
