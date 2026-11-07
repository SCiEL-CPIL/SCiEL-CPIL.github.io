import time
import random

def construct():
    n = 10
    L0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # We search for a second square L1 orthogonal to L0.
    # Constraint: For each k in 0..9, the values L1[i][j] where (i+j)%n == k must be distinct.
    
    # Backtracking search with MRV heuristic
    # Grid: 10x10
    # Variables: L1[i][j]
    # Domains: 0..9
    
    # To make it faster, we can pre-compute the diagonals.
    # diag[k] = list of (i,j) such that (i+j)%n == k
    
    diags = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            diags[(i + j) % n].append((i, j))
            
    # We will fill the grid.
    # To improve performance, we can fix the first row to be 0..9 (standard form)
    # and the first column to be 0..9? No, L1 doesn't have to be in reduced form.
    # But we can assume WLOG that L1[0][0]=0, L1[0][1]=1, etc?
    # Actually, we can just search.
    
    # Let's use a simple recursive backtracker with a time limit.
    start_time = time.time()
    timeout = 50  # seconds
    
    # State:
    # rows: list of sets of used values in each row
    # cols: list of sets of used values in each col
    # diags_used: list of sets of used values in each diagonal
    
    rows = [set() for _ in range(n)]
    cols = [set() for _ in range(n)]
    diags_used = [set() for _ in range(n)]
    grid = [[None for _ in range(n)] for _ in range(n)]
    
    # Heuristic: Order cells by constraint tightness?
    # For simplicity, we iterate row by row.
    # We can try to fix the first row to 0,1,2,3,4,5,6,7,8,9 to reduce symmetry.
    # If we fix row 0, we must ensure it's a permutation.
    
    # Let's try to fix the first row to identity permutation.
    # This is valid because we can permute columns of L1 and L0?
    # No, L0 is fixed. We can permute rows/cols of L1?
    # If we permute columns of L1, it remains a Latin square.
    # Does it remain orthogonal to L0?
    # Pairs become (L0[i][j], L1[i][sigma(j)]).
    # The set of pairs is the same, just reordered. So yes.
    # So we can assume L1[0][j] = j for all j.
    
    for j in range(n):
        grid[0][j] = j
        rows[0].add(j)
        cols[j].add(j)
        diags_used[(0 + j) % n].add(j)
        
    # Now fill the rest.
    # We need to track available values for each cell.
    
    def solve(i, j):
        if i == n:
            return True
            
        if j == n:
            return solve(i + 1, 0)
            
        if grid[i][j] is not None:
            return solve(i, j + 1)
            
        if time.time() - start_time > timeout:
            return False
            
        # Get possible values
        # Value v must not be in rows[i], cols[j], or diags_used[(i+j)%n]
        used = rows[i] | cols[j] | diags_used[(i + j) % n]
        
        # To improve search, pick a value that fails least? 
        # For simplicity, iterate 0..9.
        # We can randomize or use a specific order.
        for v in range(n):
            if v not in used:
                # Place v
                grid[i][j] = v
                rows[i].add(v)
                cols[j].add(v)
                diags_used[(i + j) % n].add(v)
                
                if solve(i, j + 1):
                    return True
                else:
                    # Undo
                    grid[i][j] = None
                    rows[i].remove(v)
                    cols[j].remove(v)
                    diags_used[(i + j) % n].remove(v)
                    
        return False

    if solve(1, 0):
        L1 = [row[:] for row in grid]
        # Verify
        # Check Latin
        valid = True
        for r in range(n):
            if set(L1[r]) != set(range(n)):
                valid = False
        for c in range(n):
            if set(L1[r][c] for r in range(n)) != set(range(n)):
                valid = False
        if valid:
            # Check Orthogonality
            pairs = set()
            for r in range(n):
                for c in range(n):
                    p = (L0[r][c], L1[r][c])
                    if p in pairs:
                        valid = False
                        break
                    pairs.add(p)
            if valid:
                return [L0, L1]
                
    return [L0]
