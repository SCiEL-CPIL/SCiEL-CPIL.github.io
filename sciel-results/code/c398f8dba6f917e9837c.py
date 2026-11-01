import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimizes the largest entry (scope).
    """
    
    # Total number of differences
    total_diffs = n_rows * k * (k + 1) // 2
    
    def check_scope(M):
        """
        Checks if a valid DTS exists with max entry <= M.
        Returns the list of rows if found, else None.
        """
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        def dfs(r, c):
            if r == n_rows:
                return True
            
            # Determine start value for x
            if c == 1:
                if r == 0:
                    start_x = 1
                else:
                    # Symmetry: enforce non-decreasing first marks
                    start_x = rows[r-1][1]
            else:
                start_x = rows[r][c-1] + 1
            
            end_x = M
            
            # Heuristic: Try smaller values first to keep scope small?
            # But we are already bounding by M.
            
            for x in range(start_x, end_x + 1):
                # Check differences
                valid = True
                new_diffs = []
                for i in range(c):
                    d = x - rows[r][i]
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    # Update
                    used_diffs.update(new_diffs)
                    rows[r].append(x)
                    
                    if c == k:
                        if dfs(r + 1, 1):
                            return True
                    else:
                        if dfs(r, c + 1):
                            return True
                    
                    # Backtrack
                    rows[r].pop()
                    used_diffs.difference_update(new_diffs)
            
            return False
        
        if dfs(0, 1):
            return [row[:] for row in rows]
        return None

    # Search for minimal M
    # Lower bound: total_diffs
    # Upper bound: total_diffs + 20 (safe margin for small n,k)
    # For n=7, k=5, total_diffs=105. Best known 112.
    
    for M in range(total_diffs, total_diffs + 30):
        result = check_scope(M)
        if result is not None:
            return result
            
    # Fallback if not found in range (should not happen for given constraints)
    # Try a larger range
    for M in range(total_diffs + 30, total_diffs + 100):
        result = check_scope(M)
        if result is not None:
            return result
            
    return None
