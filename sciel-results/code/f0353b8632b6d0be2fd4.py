import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimizes the largest entry (scope).
    """
    sys.setrecursionlimit(10000)
    
    total_diffs = n_rows * (k * (k + 1) // 2)
    lower_bound = total_diffs
    
    for M in range(lower_bound, 200):
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        def dfs(r, c, max_val):
            if r == n_rows:
                return True
            
            # Determine start_x
            if c == 1:
                if r == 0:
                    start_x = 1
                else:
                    start_x = rows[r-1][1]
            else:
                start_x = rows[r][c-1] + 1
            
            end_x = max_val + 1
            
            for x in range(start_x, end_x):
                # Symmetry breaking: If r > 0 and rows[r-1][1] == x (and c==1),
                # we must ensure rows[r] is lexicographically > rows[r-1].
                # If c > 1, we are already in the row.
                
                # Check differences
                valid = True
                new_diffs = []
                for i in range(c):
                    d = x - rows[r][i]
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue

                # Additional symmetry check for c==1
                if c == 1 and r > 0 and x == rows[r-1][1]:
                    # We are starting a row with the same first element as the previous row.
                    # We must ensure the rest of this row is lexicographically larger.
                    # Since we haven't filled the rest, we can't check fully yet.
                    # However, we can enforce that if we are at c==1, and x == prev_first,
                    # we must pick a value for c==2 that is > prev_row[2] if prev_row[2] exists?
                    # This is complex to check inline.
                    # Simpler: Just enforce rows[r] > rows[r-1] lexicographically.
                    # We can check this when we complete the row or at each step.
                    # At step c, if rows[r][:c+1] == rows[r-1][:c+1], then rows[r][c+1] MUST be > rows[r-1][c+1] (if it exists).
                    # If c+1 > k, then rows[r] == rows[r-1], which is invalid (duplicate rows not allowed? 
                    # Actually, duplicate rows would have identical differences, so they are invalid anyway 
                    # because differences must be distinct over all rows. 
                    # Wait, if row A == row B, then diffs(A) == diffs(B). 
                    # The set of all diffs is union. If they are identical, they are not distinct.
                    # So duplicate rows are impossible in a valid solution.
                    # Thus, we just need to ensure we don't generate duplicate rows.
                    # The lexicographic ordering handles this.
                    
                    # If x == rows[r-1][1], we are on the "same prefix".
                    # We must ensure that eventually rows[r] > rows[r-1].
                    # We can prune if at any point c <= k and rows[r][:c+1] == rows[r-1][:c+1] 
                    # and c == k (rows are identical) -> invalid.
                    # Or if c < k and rows[r][c] == rows[r-1][c], we must pick rows[r][c+1] > rows[r-1][c+1].
                    
                    # For now, let's rely on the fact that differences must be unique.
                    # If rows are identical, diffs are identical, so conflict.
                    # The `used_diffs` check will catch conflicts within the row.
                    # But it won't catch conflicts *between* identical rows until we try to add the diffs of the second row.
                    # When we add row r, we check its diffs against `used_diffs` which contains diffs of rows 0..r-1.
                    # So if row r is identical to row r-1, its diffs are already in `used_diffs`.
                    # So the first diff of row r will conflict.
                    # So we don't need explicit lexicographic check for correctness, only for speed.
                    # However, lexicographic ordering reduces the search space by factor of N!.
                    
                    # Let's add a simple check: if c==1 and r>0 and x < rows[r-1][1], skip (handled by start_x).
                    # If c==1 and r>0 and x == rows[r-1][1], we allow it, but the diff check will fail if the row turns out identical.
                    # To prune earlier: if c < k and x == rows[r-1][c], we *must* have next value > rows[r-1][c+1].
                    # We can enforce this by adjusting the start_x for the *next* step?
                    # Or just let the diff check handle it. It's efficient enough for N=7.
                    
                    pass
                
                # Add to used
                used_diffs.update(new_diffs)
                rows[r].append(x)
                
                if c == k:
                    if dfs(r + 1, 1, max_val):
                        return True
                else:
                    if dfs(r, c + 1, max_val):
                        return True
                
                # Backtrack
                rows[r].pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        if dfs(0, 1, M):
            return rows
            
    return rows
