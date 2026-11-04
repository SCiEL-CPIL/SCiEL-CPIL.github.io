"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    # Block decomposition splitting all three dimensions:
    # A (3x3): rows 2+1, cols 2+1
    # B (3x4): rows 2+1  
    # C (3x4): rows 2+1
    # Sub-problems: A11(2x2)*B1(2x4) rank 8, A12(2x1)*B2(1x4) rank 8,
    #               A21(1x2)*B1(2x4) rank 8, A22(1x1)*B2(1x4) rank 4
    # Total rank: 28
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # A11*B1: i in 0-1, j in 0-1, k in 0-3
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    # A12*B2: i in 0-1, j=2, k in 0-3
    for i in range(2):
        for k in range(4):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    # A21*B1: i=2, j in 0-1, k in 0-3
    for j in range(2):
        for k in range(4):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    # A22*B2: i=2, j=2, k in 0-3
    for k in range(4):
        U[2*m + 2][col] = 1
        V[2*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
