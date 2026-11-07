import time
import random

def construct():
    n = 10
    # L1 is the standard cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Target: Find L2 orthogonal to L1
    # We will use backtracking to fill L2
    # L2[i][j] must be in 0..n-1
    # Row i must contain all 0..n-1
    # Col j must contain all 0..n-1
    # Pair (L1[i][j], L2[i][j]) must be unique across all (i,j)
    
    deadline = time.time() + 45  # 45s budget for search, leaving 45s safety margin
    
    # Pre-compute constraints for L2
    # For each cell (i,j), possible values are 0..n-1
    # We maintain:
    # row_used[i]: set of values used in row i
    # col_used[j]: set of values used in col j
    # pair_used: set of tuples (L1[i][j], val)
    
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pair_used = set()
    
    # Order of cells to fill: 
    # We can iterate row by row.
    # To speed up, we can use MRV (Minimum Remaining Values)
    # But for simplicity, let's try row-major with forward checking.
    
    def is_valid(i, j, val):
        if val in row_used[i]: return False
        if val in col_used[j]: return False
        if (L1[i][j], val) in pair_used: return False
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n * n:
            return True
        
        i = idx // n
        j = idx % n
        
        # Heuristic: Try values that are most constrained?
        # For now, try 0..n-1
        # Optimization: Shuffle values to avoid bias?
        vals = list(range(n))
        random.shuffle(vals)
        
        for val in vals:
            if is_valid(i, j, val):
                L2[i][j] = val
                row_used[i].add(val)
                col_used[j].add(val)
                pair_used.add((L1[i][j], val))
                
                if backtrack(idx + 1):
                    return True
                else:
                    # Undo
                    L2[i][j] = -1
                    row_used[i].remove(val)
                    col_used[j].remove(val)
                    pair_used.remove((L1[i][j], val))
        
        return False

    # Start search
    # To improve success rate, we can restart with different shuffles if it fails quickly
    # But simple backtracking might work.
    
    found = backtrack(0)
    
    if found:
        # Verify
        # Check L2 is Latin
        for i in range(n):
            if sorted(L2[i]) != list(range(n)):
                return [L1]
            if sorted([L2[i][j] for i in range(n)]) != list(range(n)):
                return [L1]
        # Check Orthogonality
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (L1[i][j], L2[i][j])
                if p in pairs:
                    return [L1]
                pairs.add(p)
        return [L1, L2]
    else:
        return [L1]
