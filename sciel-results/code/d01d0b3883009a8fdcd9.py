# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    n = 10
    # Construct L1
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find L2 orthogonal to L1
    L2 = _find_mols(L1, n, start_time)
    
    if L2 is not None:
        return [L1, L2]
    else:
        return [L1]

def _find_mols(L1, n, start_time):
    # Backtracking to find L2
    # L2[i][j] must be 0..n-1
    # Constraints:
    # 1. Row i: distinct
    # 2. Col j: distinct
    # 3. Orthogonality: for each k, L2[i][j] where (L1[i][j] == k) are distinct.
    
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Precompute orthogonality classes (diagonals)
    # diag[k] = list of (i, j) such that L1[i][j] == k
    diags = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            k = L1[i][j]
            diags[k].append((i, j))
            
    # Helper to check if placing val at (i, j) is valid
    def is_valid(i, j, val):
        # Check row
        for jj in range(n):
            if grid[i][jj] == val:
                return False
        # Check col
        for ii in range(n):
            if grid[ii][j] == val:
                return False
        # Check orthogonality class
        k = L1[i][j]
        for (ii, jj) in diags[k]:
            if grid[ii][jj] == val:
                return False
        return True

    # Find next empty cell
    def find_empty():
        # Simple heuristic: first empty cell
        # Better: most constrained
        for i in range(n):
            for j in range(n):
                if grid[i][j] == -1:
                    return i, j
        return -1, -1

    def backtrack():
        if time.time() - start_time > 50: # Safety margin
            return False
        
        i, j = find_empty()
        if i == -1:
            return True # Success
            
        for val in range(n):
            if is_valid(i, j, val):
                grid[i][j] = val
                if backtrack():
                    return True
                grid[i][j] = -1
        return False

    if backtrack():
        return grid
    return None

# EVOLVE-BLOCK-END
