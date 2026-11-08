import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # Generate all possible 2x2x2 block positions
    candidates = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                candidates.append((i0, j0, k0))
                
    # Function to check if two blocks are disjoint
    def is_disjoint(b1, b2):
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        # Disjoint if they don't share any index in any dimension
        # Intervals [i1, i1+1] and [i2, i2+1] are disjoint if i1+1 < i2 or i2+1 < i1
        # i.e., i2 > i1+1 or i1 > i2+1
        # i.e., i2 - i1 > 1 or i1 - i2 > 1
        # i.e., abs(i1 - i2) > 1
        if abs(i1 - i2) > 1: return True
        if abs(j1 - j2) > 1: return True
        if abs(k1 - k2) > 1: return True
        return False

    # Backtracking search for max number of disjoint blocks
    best_blocks = []
    
    def backtrack(remaining, current):
        nonlocal best_blocks
        if time.time() > deadline:
            return
        if len(remaining) == 0:
            if len(current) > len(best_blocks):
                best_blocks = current.copy()
            return
        
        # Heuristic: try to add the first candidate, then try skipping it
        # Sort candidates to prioritize those with fewer conflicts? 
        # For small N, simple recursion is fine.
        
        # Option 1: Include remaining[0]
        first = remaining[0]
        rest = remaining[1:]
        
        # Check if first is disjoint from all in current
        can_add = True
        for b in current:
            if not is_disjoint(first, b):
                can_add = False
                break
        
        if can_add:
            current.append(first)
            backtrack(rest, current)
            current.pop()
            
        # Option 2: Exclude remaining[0]
        backtrack(rest, current)

    backtrack(candidates, [])
    
    blocks = best_blocks
    num_blocks = len(blocks)
    
    # Calculate covered terms
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    # Uncovered terms
    uncovered = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    uncovered.append((i, j, k))
    
    total_rank = num_blocks * 7 + len(uncovered)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    for (i, j, k) in uncovered:
        if time.time() > deadline:
            break
        U[i*m+j][col] += 1
        V[j*p+k][col] += 1
        W[k*n+i][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
