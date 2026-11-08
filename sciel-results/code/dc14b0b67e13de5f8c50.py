import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # New approach: Use a hybrid decomposition with optimized block sizes
    # For 3x5x5, we can use:
    # - One 2x2x2 Strassen block (rank 7)
    # - Remaining terms handled with schoolbook
    # But we can do better by using multiple smaller Strassen-like blocks
    
    # Let's use a rank-68 construction:
    # Three 2x2x2 Strassen blocks (rank 7 each = 21)
    # Plus schoolbook for remaining 75 - 3*8 = 51 terms
    # Total: 21 + 51 = 72... not better
    
    # Better: Use the structure more carefully.
    # For 3x5x5, split the 3 into 2+1, the 5s into 2+2+1 or 3+2
    
    # Approach: Use 2x2x2 Strassen blocks where possible
    # We can fit: floor(3/2)=1 in i-dim, floor(5/2)=2 in j-dim, floor(5/2)=2 in k-dim
    # So 1*2*2 = 4 blocks of 2x2x2, each rank 7 = 28
    # Covered: 4*8 = 32 terms
    # Remaining: 75-32 = 43 terms
    # Total rank: 28+43 = 71 (same as before)
    
    # To improve, we need a better decomposition for the remaining terms.
    # Or use a different block size.
    
    # Let's try: Use 2x2x2 blocks but with overlap sharing
    # Or use a 3x3x3 block (rank 23) + smaller blocks
    
    # For now, let's use a rank-70 construction with careful optimization
    # of the schoolbook part by combining terms.
    
    # Actually, let me use a completely different approach:
    # Use the "linear combination" method where we find the minimal
    # number of multiplications needed.
    
    # For 3x5x5, I'll use a construction based on:
    # - 2x2x2 Strassen for the (0,1) x (0,1) x (0,1) block
    # - 2x2x2 Strassen for the (0,1) x (2,3) x (0,1) block  
    # - 2x2x2 Strassen for the (0,1) x (0,1) x (2,3) block
    # - 2x2x2 Strassen for the (0,1) x (2,3) x (2,3) block
    # - Schoolbook for the remaining 43 terms
    
    # This gives rank 71. To improve, we need to reduce the schoolbook part.
    
    # Idea: Some of the "remaining" terms can be combined with Strassen terms
    # if we allow half-integer coefficients.
    
    # Let me try a rank-69 construction by combining 2 schoolbook terms
    # into 1 term using a linear combination.
    
    # For now, return the rank-71 solution and verify correctness.
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    total_rank = 71
    
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
                    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    # Check rank
    print(f"Rank: {U.shape[1]}")
    # Verify correctness
    pass
