import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    DEADLINE = 110.0  # safety margin
    
    # Start with the known rank-71 construction (4 Strassen blocks + schoolbook)
    # Then try to find a better decomposition using algebraic optimization
    
    # First, build the rank-71 baseline
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7  # 28
    
    total_terms = n * m * p  # 75
    strassen_terms_count = num_strassen_blocks * 8  # 32
    schoolbook_terms_count = total_terms - strassen_terms_count  # 43
    
    total_rank = total_strassen_rank + schoolbook_terms_count  # 71
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        """Add Strassen rank-7 decomposition for a 2x2x2 block."""
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
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Now try to reduce rank by finding linear dependencies among the rank-1 terms
    # We can try to merge some of the schoolbook terms with Strassen terms
    # or find that some combinations are redundant
    
    # Alternative approach: try a different block structure
    # What if we use 1x1x1 blocks for row 2 (i=2) and Strassen for rows 0,1?
    
    # Actually, let's try to find a better partition
    # The key insight: we have 3 rows in A, 5 columns in A, 5 rows in B, 5 columns in B
    # We can try different 2x2 blocks
    
    # Try: use Strassen on (i=0,1; j=0,1; k=0,1) and (i=0,1; j=2,3; k=0,1)
    # and (i=0,1; j=0,1; k=2,3) and (i=0,1; j=2,3; k=2,3)
    # This is what we already have.
    
    # What about trying to cover more with Strassen?
    # We have row i=2 not covered. Can we do a 3x2x2 block? No, Strassen is for 2x2x2.
    
    # Let's try a different approach: use a 3x3x3 block for the top-left
    # But we only have 3x5x5, so we can do a 3x3x3 block on i=0..2, j=0..2, k=0..2
    # Rank of 3x3x3 matrix multiplication is known to be <= 23 (Laderman)
    # But implementing Laderman is complex.
    
    # Alternative: try to optimize the existing rank-71 by checking if any
    # linear combination of the rank-1 tensors can reduce the rank.
    
    # For now, let's try a slightly different block arrangement that might
    # allow better merging.
    
    # Try: 2 Strassen blocks for i=0,1 and handle i=2 separately with schoolbook
    # But also try to use a 2x5x5 block for i=0,1 (two 2x2x2 blocks in j,k)
    
    # Actually, let's try yet another approach: 
    # Use 1 Strassen block for (i=0,1; j=0,1; k=0,1)
    # And schoolbook for the rest: 75 - 8 = 67 terms
    # Total rank: 7 + 67 = 74. Worse than 71.
    
    # The current 71 is likely near-optimal for this simple approach.
    
    # Let's try to verify and return
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(total_rank))
                expected = 1.0
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to pure schoolbook
        total_rank_fb = n * m * p
        U_fb = [[0.0] * total_rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * total_rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * total_rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1.0
                    V_fb[j*p+k][col_fb] = 1.0
                    W_fb[k*n+i][col_fb] = 1.0
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
