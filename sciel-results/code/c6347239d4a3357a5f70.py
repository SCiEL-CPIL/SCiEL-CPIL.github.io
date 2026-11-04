"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Rank 68 decomposition:
    # 64 terms for k=0..3 (schoolbook for 4x4x4)
    # 4 terms for k=4 (schoolbook for 4x4x1)
    
    rank = 64 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    # Part 1: k = 0 to 3
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Part 2: k = 4
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # This is decomposed into 4 rank-1 terms, one for each j.
    for j in range(m):
        r = column
        # U[i*m + j, r] = 1 for all i
        for i in range(n):
            U[i * m + j][r] = 1
        # V[j*p + 4, r] = 1
        V[j * p + 4][r] = 1
        # W[4*n + i, r] = 1 for all i
        for i in range(n):
            W[4 * n + i][r] = 1
        column += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
