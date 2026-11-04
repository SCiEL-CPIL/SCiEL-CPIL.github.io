"""Difference triangle set (7,5): minimize the largest entry."""
import time

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Find n_rows of k+1 integers starting at 0, strictly increasing,
    such that all pairwise differences are distinct across all rows.
    Minimizes the largest integer used.
    """
    # The counting bound for n=7, k=5 is 105.
    # We search for the smallest M such that a valid configuration exists in [0, M].
    
    def _check_and_find(M):
        """
        Returns a valid set of rows if one exists with max value <= M, else None.
        Uses backtracking with forward checking.
        """
        used_diffs = set()
        rows = []
        # For symmetry breaking, we can enforce rows are sorted by their max element.
        # i.e., rows[0][-1] <= rows[1][-1] <= ...
        
        def _build_row(row_idx, current_row, row_diffs, min_last_val):
            """
            Recursively build the row at index row_idx.
            current_row: list of ints already in this row.
            row_diffs: set of differences within current_row.
            min_last_val: the last element of this row must be >= this value (symmetry breaking).
            """
            if len(current_row) == k + 1:
                # Row is complete. Check if it satisfies symmetry breaking.
                if current_row[-1] < min_last_val:
                    return False
                
                # Commit this row
                rows.append(list(current_row))
                used_diffs.update(row_diffs)
                
                if row_idx == n_rows - 1:
                    return True
                
                # Recurse to next row
                # The next row's max must be >= current row's max
                if _build_row(row_idx + 1, [0], set(), current_row[-1]):
                    return True
                
                # Backtrack
                rows.pop()
                used_diffs.difference_update(row_diffs)
                return False
            
            # Determine next value
            start_val = current_row[-1] + 1 if current_row else 0
            # We don't know the final max yet, but we know it must be <= M.
            # Also, if we are at the last element, it must be >= min_last_val.
            # To prune early, if we are at the last element (index k), we must pick x such that x >= min_last_val.
            
            # Heuristic: try smaller values first to keep the row "compact", 
            # but we also need to ensure we can reach min_last_val if required.
            
            # Optimization: If we are at the last element, we must pick x >= min_last_val.
            # If we are not at the last element, we just need to ensure that 
            # it is *possible* to reach min_last_val with the remaining elements.
            # This is complex, so we just iterate and let the recursion handle it.
            
            for x in range(start_val, M + 1):
                # Check if adding x creates conflicts
                new_diffs = set()
                for a in current_row:
                    d = x - a
                    new_diffs.add(d)
                
                # Conflict with global used diffs
                if new_diffs & used_diffs:
                    continue
                
                # Conflict with internal row diffs
                if new_diffs & row_diffs:
                    continue
                
                # Additional pruning: 
                # If this is the last element, check symmetry constraint
                if len(current_row) == k: # next is the (k+1)-th element, index k
                    if x < min_last_val:
                        continue
                
                # Also, we must ensure that even if we pick x, we can still 
                # potentially satisfy the symmetry constraint for the LAST element 
                # if we haven't reached it yet. 
                # Actually, the symmetry constraint is on the LAST element of the row.
                # So if we are not at the last element, we don't strictly need to check min_last_val yet, 
                # but we should ensure that x is not so large that we overshoot M? No, x <= M is enforced.
                
                # Recurse
                current_row.append(x)
                row_diffs.update(new_diffs)
                
                if _build_row(row_idx, current_row, row_diffs, min_last_val):
                    return True
                
                current_row.pop()
                row_diffs.difference_update(new_diffs)
                
            return False
        
        # Start building the first row
        # Symmetry breaking for the first row: min_last_val = 0
        if _build_row(0, [0], set(), 0):
            return rows
        return None

    # Search for the optimal M
    # Start from the known lower bound
    # The simple counting bound is 105.
    # The best known is 112.
    # We will search from 105 upwards.
    
    # To be safe and fast, we can start a bit higher if 105-111 are unlikely, 
    # but backtracking is fast enough to check them.
    
    for M in range(105, 150):
        # Add a time check to avoid hanging if something goes wrong, 
        # though this should be very fast.
        if time.time() > 140:
            break
            
        result = _check_and_find(M)
        if result is not None:
            return result
            
    # Fallback: if no solution found in range (should not happen), 
    # return a greedy construction to ensure validity.
    # This is a safety net.
    return _greedy_fallback(n_rows, k)

def _greedy_fallback(n_rows, k):
    """Simple greedy construction to ensure a valid output is always returned."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
