"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    
    Uses a backtracking search with dynamic variable ordering (MRV) over the
    difference constraint satisfaction problem.
    """
    # Total number of differences needed: n_rows * k*(k+1)/2
    # Each row has k+1 entries, so C(k+1, 2) = k*(k+1)/2 differences per row
    total_diffs = n_rows * k * (k + 1) // 2
    
    # We need to find the minimal M such that we can place all differences in [1, M]
    # with the structural constraints of difference triangles.
    
    # First, try the greedy approach to get an upper bound
    greedy_result = _greedy_construct(n_rows, k)
    if greedy_result is not None:
        best_max = max(max(row) for row in greedy_result)
    else:
        best_max = 200  # fallback upper bound
    
    # Try to improve with backtracking for small instances
    # For larger instances, just return the greedy result
    
    if n_rows * k * (k + 1) // 2 > 500:
        # Too many differences for backtracking to be feasible
        return greedy_result
    
    # Try backtracking to find a better solution
    bt_result = _backtracking_construct(n_rows, k, best_max)
    if bt_result is not None:
        return bt_result
    
    return greedy_result


def _greedy_construct(n_rows, k):
    """Greedy construction: each row starts at 0 and takes the smallest next mark
    whose differences to the earlier marks of the row are all still unused by any row."""
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


def _backtracking_construct(n_rows, k, max_bound):
    """
    Backtracking search with MRV heuristic.
    We build rows one at a time, but use MRV to decide which difference to place next.
    
    For each row, we need to choose k+1 positions p_0=0 < p_1 < ... < p_k such that
    all differences p_j - p_i for i<j are in the unused set and distinct.
    
    We represent the state as:
    - used: set of used differences
    - current_row: list of positions chosen so far for the current row
    
    The MRV heuristic: for the current row, consider all possible next positions x
    that haven't been tried. For each candidate x, compute the set of new differences
    {x - a for a in current_row}. The "constraint" is the number of these differences
    that are already used (should be 0) and the number of new differences that would
    conflict with other potential placements.
    
    A simpler approach: for each row, we do a DFS with MRV on the differences.
    """
    used = set()
    rows = []
    
    def backtrack(row_idx, current_row, max_val):
        if row_idx == n_rows:
            return True, max_val
        
        # Try to build the next row
        if _build_row(current_row, used, k, max_val, rows, row_idx):
            return True, max_val
        
        return False, max_val
    
    def _build_row(row, used, k, max_val, rows, row_idx):
        """
        Build a row of k+1 elements starting with [0], using backtracking with MRV.
        row is the current partial row (starts with [0]).
        Returns True if a complete row was found and added to rows.
        """
        if len(row) == k + 1:
            # Check all differences are valid
            diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used or d in diffs:
                        return False
                    diffs.add(d)
            # All good, add to used and rows
            used.update(diffs)
            rows.append(row[:])
            return True
        
        # MRV: find the best next position to try
        # For each candidate position x > row[-1], compute the new differences
        # and count how many are "free" (not in used and not conflicting with each other)
        
        best_x = None
        best_score = -1
        
        # We need to find a position x such that all differences x - a for a in row
        # are not in used and are distinct from each other
        # Since row is sorted, x - a for a in row are all distinct if x is fixed
        
        # Try candidates in order of increasing x, but use MRV to prioritize
        # candidates that have the fewest "free slots" for future placements
        
        # Simple approach: try x from row[-1]+1 to max_val
        # For MRV, we can look ahead: for each candidate x, after placing it,
        # how many options do we have for the next position?
        
        # For efficiency, just try sequentially but with pruning
        start = row[-1] + 1 if row else 1
        for x in range(start, max_val + 1):
            # Check if x - a is in used for any a in row
            valid = True
            new_diffs = set()
            for a in row:
                d = x - a
                if d in used:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Place x and recurse
            row.append(x)
            # Temporarily mark new_diffs as used for this branch
            old_used = used.copy()
            used.update(new_diffs)
            
            if _build_row(row, used, k, max_val, rows, row_idx):
                return True
            
            # Backtrack
            used.clear()
            used.update(old_used)
            row.pop()
        
        return False
    
    # Try with the current best_max as the bound
    rows.clear()
    used.clear()
    result, _ = backtrack(0, [0], best_max)
    
    if result:
        return rows
    
    return None
# EVOLVE-BLOCK-END
