import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    n_rows: number of rows
    k: number of differences per row (row length is k+1)
    """
    
    # Deadline guard: 120s limit, use 110s to be safe
    deadline = time.time() + 110
    
    # 1. Greedy Initial Solution
    def greedy_construct():
        used = 0  # bitmask of used differences
        rows = []
        for _ in range(n_rows):
            row = [0]
            # We need k more elements
            # Greedy: pick smallest x > last such that diffs are free
            # We don't know M in advance, so we just grow.
            # To keep it tight, we can estimate M or just let it grow.
            # Standard greedy for Golomb rulers tends to be reasonable.
            for _ in range(k):
                last = row[-1]
                x = last + 1
                while True:
                    # Check differences x - a for a in row
                    conflict = False
                    mask = 0
                    for a in row:
                        d = x - a
                        if d < 0: continue # x > a always
                        if (used >> d) & 1:
                            conflict = True
                            break
                        mask |= (1 << d)
                    if not conflict and (mask & used) == 0:
                        # Also check internal conflicts? 
                        # Diffs x-a are distinct if a are distinct.
                        # So just check against global used.
                        row.append(x)
                        used |= mask
                        break
                    x += 1
            rows.append(row)
        return rows, used

    best_rows, _ = greedy_construct()
    best_M = max(max(r) for r in best_rows)
    
    # Lower bound estimate
    # Counting bound: N * k(k+1)/2 <= M(M+1)/2 approx
    # Total diffs = n_rows * k(k+1)/2
    # Max diff <= M
    # So M(M+1)/2 >= n_rows * k(k+1)/2
    # M ~ sqrt(n_rows * k^2)
    total_diffs = n_rows * k * (k + 1) // 2
    # M(M+1)/2 >= total_diffs
    # M^2 ~ 2 * total_diffs
    import math
    lower_bound = int(math.ceil((math.sqrt(8 * total_diffs + 1) - 1) / 2))
    # Add a small margin to lower bound
    lower_bound = max(0, lower_bound - 5)
    
    # 2. Iterative Deepening / Downward Search
    # Try to find solutions with smaller M
    current_M = best_M - 1
    
    def can_construct_with_bound(M, global_used_mask=0, rows_needed=n_rows):
        """
        Attempts to construct 'rows_needed' rows with max entry <= M.
        Uses backtracking for each row.
        Returns a list of rows if successful, else None.
        """
        if rows_needed == 0:
            return []
        
        # Heuristic: Construct the "hardest" row first? 
        # Usually, later rows are harder as more diffs are used.
        # So we construct rows sequentially.
        
        # We will recursively build the first row, then call this function for the rest.
        # To avoid deep recursion overhead in Python for many rows, we can just loop rows 
        # and use a helper to build a single row given the current global mask.
        
        current_rows = []
        mask = global_used_mask
        
        for _ in range(rows_needed):
            row = build_single_row(mask, M, k, deadline)
            if row is None:
                return None
            current_rows.append(row)
            # Update mask with new diffs
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    mask |= (1 << d)
        
        return current_rows

    def build_single_row(global_mask, M, k, deadline):
        """
        Builds a single row of length k+1 starting at 0, max entry <= M.
        Uses backtracking.
        """
        if time.time() > deadline:
            return None
        
        row = [0]
        # We need to add k elements.
        
        def backtrack(idx, current_row, local_mask):
            if time.time() > deadline:
                return False
            
            if idx == k + 1:
                return True
            
            last = current_row[-1]
            # Try next values x
            # Optimization: We can restrict x range.
            # x must be > last.
            # x <= M.
            
            # Heuristic: Try smaller x first? Or larger?
            # Smaller x keeps M small, which is good for the global bound.
            for x in range(last + 1, M + 1):
                # Check conflicts with current_row
                conflict = False
                new_mask = 0
                for a in current_row:
                    d = x - a
                    if d < 0: continue
                    if (global_mask >> d) & 1 or (local_mask >> d) & 1:
                        conflict = True
                        break
                    new_mask |= (1 << d)
                
                if not conflict:
                    # Also check if new_mask has internal conflicts?
                    # Diffs x-a are distinct for distinct a. So no internal conflict.
                    # Just need to ensure (new_mask & (global_mask | local_mask)) == 0
                    # Which is covered by the loop check.
                    
                    current_row.append(x)
                    if backtrack(idx + 1, current_row, local_mask | new_mask):
                        return True
                    current_row.pop()
                    
                    # Pruning: If we failed with x, maybe larger x will work?
                    # Yes, so we continue the loop.
                    
            return False
        
        # Start backtracking
        # We can add a simple forward check: 
        # Is there enough room?
        # Not strictly necessary for small k, but helps.
        
        if backtrack(0, row, 0):
            return row
        return None

    # Search loop
    while current_M >= lower_bound and time.time() < deadline:
        # Try to construct with this M
        # We need to reset global_mask to 0 for each new M attempt
        result = can_construct_with_bound(current_M)
        if result is not None:
            # Valid solution found
            best_rows = result
            best_M = current_M
            # We found a smaller M, try to go even smaller
            current_M -= 1
        else:
            # No solution for this M, try smaller M?
            # Wait, if no solution for M, there is no solution for M-1.
            # So we should stop?
            # The property is: If a solution exists for M, it exists for M+1.
            # So if NO solution for M, then NO solution for any M' < M.
            # Therefore, the minimal M is the first one we find when searching UPWARDS.
            # OR the last one we find when searching DOWNWARDS from a known valid M.
            
            # My loop starts at best_M - 1 (which is valid? No, best_M is valid, best_M-1 might not be).
            # If best_M - 1 fails, then best_M is the minimum?
            # Not necessarily. It's possible that best_M - 1 fails, but best_M - 2 works? No.
            # If it fails for M, it fails for all smaller.
            # So if `can_construct_with_bound(current_M)` returns None, we should STOP.
            # Because we are looking for the MINIMUM M.
            # Since we started from a valid M (best_M) and went down, the first valid one we hit is the minimum.
            # If we hit an invalid one, all smaller are invalid.
            
            # Wait, is `can_construct_with_bound` complete?
            # It uses backtracking, so it is complete (if it finishes).
            # So if it returns None, no solution exists for that M.
            
            break

    return best_rows
