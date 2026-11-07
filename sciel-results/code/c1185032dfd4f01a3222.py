import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n = 10
    # Deadline guard
    start_time = time.time()
    deadline = 60.0 # Safety margin before 90s kill
    
    # Square 1: Cyclic
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # We want to find L2 orthogonal to L1
    # L2 is a list of lists of None initially
    L2 = [[None for _ in range(n)] for _ in range(n)]
    
    # Sets for constraint checking
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pair_used = set()
    
    # Precompute L1 values
    # We will fill L2 cell by cell
    
    def is_valid(i, j, v):
        if v in row_used[i]:
            return False
        if v in col_used[j]:
            return False
        if (L1[i][j], v) in pair_used:
            return False
        return True

    def solve(i, j):
        # Check time
        if time.time() > deadline:
            return False
        
        # Find next empty cell using MRV heuristic
        min_domain = None
        min_size = n + 1
        min_i, min_j = -1, -1
        
        for r in range(n):
            for c in range(n):
                if L2[r][c] is None:
                    # Count valid values
                    valid_vals = []
                    for v in range(n):
                        if is_valid(r, c, v):
                            valid_vals.append(v)
                    if len(valid_vals) == 0:
                        return False # Dead end
                    if len(valid_vals) < min_size:
                        min_size = len(valid_vals)
                        min_domain = valid_vals
                        min_i, min_j = r, c
                        if min_size == 1:
                            break
            if min_size == 1:
                break
        
        if min_i == -1:
            # All cells filled
            return True
        
        # Try values for the selected cell
        for v in min_domain:
            # Place
            L2[min_i][min_j] = v
            row_used[min_i].add(v)
            col_used[min_j].add(v)
            pair_used.add((L1[min_i][min_j], v))
            
            if solve(min_i, min_j):
                return True
            
            # Unplace
            L2[min_i][min_j] = None
            row_used[min_i].remove(v)
            col_used[min_j].remove(v)
            pair_used.remove((L1[min_i][min_j], v))
        
        return False

    if solve(0, 0):
        return [L1, L2]
    else:
        # If search fails or times out, return just L1
        return [L1]

# EVOLVE-BLOCK-END
