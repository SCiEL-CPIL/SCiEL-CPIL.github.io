"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import sys
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry (the scope).
    
    Uses a CSP-based backtracking with MRV (Minimum Remaining Values) heuristic.
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    total_diffs = n_rows * (k * (k + 1) // 2)
    start_m = total_diffs
    
    for attempt in range(50):
        current_m = start_m + attempt
        result = _csp_construct(n_rows, k, current_m)
        if result is not None:
            return result
    
    return _greedy_construct(n_rows, k)


def _csp_construct(n_rows, k, max_val):
    """CSP backtracking with MRV."""
    # rows[r][p]: value at position p in row r, or None if unassigned
    rows = [[0] + [None] * k for _ in range(n_rows)]
    used_diffs = set()
    
    # Unassigned variables: (r, p) for r in range(n_rows), p in 1..k
    # We'll use recursive backtracking with MRV
    
    def get_domain(r, p):
        """Get valid values for position (r, p)."""
        if p == 0:
            return [0]
        
        lower = rows[r][p-1] + 1 if rows[r][p-1] is not None else 1
        upper = max_val
        if p < k and rows[r][p+1] is not None:
            upper = rows[r][p+1] - 1
        
        if upper < lower:
            return []
        
        domain = []
        for v in range(lower, upper + 1):
            valid = True
            for i in range(p):
                d = v - rows[r][i]
                if d in used_diffs:
                    valid = False
                    break
            if not valid:
                continue
            for i in range(p + 1, k + 1):
                if rows[r][i] is not None:
                    d = rows[r][i] - v
                    if d in used_diffs:
                        valid = False
                        break
            if valid:
                domain.append(v)
        
        return domain
    
    def search(depth):
        if depth == n_rows * k:
            return True
        
        # MRV: find unassigned (r, p) with smallest domain
        best_r, best_p = -1, -1
        best_domain = None
        best_size = float('inf')
        
        for r in range(n_rows):
            for p in range(1, k + 1):
                if rows[r][p] is None:
                    domain = get_domain(r, p)
                    if len(domain) == 0:
                        return False
                    if len(domain) < best_size:
                        best_size = len(domain)
                        best_r, best_p = r, p
                        best_domain = domain
        
        if best_r == -1:
            return True  # All assigned
        
        for v in best_domain:
            rows[best_r][best_p] = v
            
            # Add new differences
            new_diffs = []
            for i in range(best_p):
                new_diffs.append(v - rows[best_r][i])
            for i in range(best_p + 1, k + 1):
                if rows[best_r][i] is not None:
                    new_diffs.append(rows[best_r][i] - v)
            
            used_diffs.update(new_diffs)
            
            if search(depth + 1):
                return True
            
            used_diffs.difference_update(new_diffs)
            rows[best_r][best_p] = None
        
        return False
    
    if search(0):
        return rows
    return None


def _greedy_construct(n_rows, k):
    """Greedy fallback."""
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

