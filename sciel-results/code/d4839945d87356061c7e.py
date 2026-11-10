"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry (scope).
    """
    import sys
    sys.setrecursionlimit(20000)
    
    total_diffs_per_row = k * (k + 1) // 2
    
    def greedy_construct():
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
    
    greedy_rows = greedy_construct()
    upper = max(max(row) for row in greedy_rows)
    lower = total_diffs_per_row * n_rows  # minimum distinct diffs needed
    
    def check_feasible(max_val):
        """
        Check if we can pack n_rows rows with marks in [0, max_val] and all differences distinct.
        Returns the list of rows if feasible, else None.
        """
        used = set()
        rows = []
        
        def backtrack(row_idx):
            if row_idx == n_rows:
                return True
            
            # Try to find a valid row for this index
            # Use a helper to find marks
            result = find_row(used, max_val, k)
            if result is None:
                return False
            
            marks, diffs = result
            rows.append(marks)
            used.update(diffs)
            
            if backtrack(row_idx + 1):
                return True
            
            rows.pop()
            used.difference_update(diffs)
            return False
        
        def find_row(used_set, max_val, k):
            """Find k+1 marks starting at 0 with differences disjoint from used_set."""
            marks = [0]
            diffs = set()
            
            def _backtrack():
                if len(marks) == k + 1:
                    return True
                last = marks[-1]
                for x in range(last + 1, max_val + 1):
                    new_diffs = {x - a for a in marks}
                    if new_diffs & used_set or new_diffs & diffs:
                        continue
                    marks.append(x)
                    diffs.update(new_diffs)
                    if _backtrack():
                        return True
                    marks.pop()
                    diffs.difference_update(new_diffs)
                return False
            
            if _backtrack():
                return list(marks), set(diffs)
            return None
        
        if backtrack(0):
            return rows
        return None
    
    # Binary search on max_val
    lo, hi = lower, upper
    best_rows = greedy_rows
    best_scope = upper
    
    while lo <= hi:
        mid = (lo + hi) // 2
        result = check_feasible(mid)
        if result is not None:
            best_rows = result
            best_scope = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_rows

# EVOLVE-BLOCK-END
