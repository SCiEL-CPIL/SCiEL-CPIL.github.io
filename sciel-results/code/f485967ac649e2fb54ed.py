import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # Core 4x4x4 (indices 0-3): Schoolbook rank 64
    # C[0:4, 4] (4x1 right border): Rank 16 (4 rows, 4 cols of A/B involved)
    # C[4, 0:4] (1x4 bottom border): Rank 16
    # C[4, 4] (corner): Rank 1
    # Total Rank = 64 + 16 + 16 + 1 = 97
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Core 4x4x4: Schoolbook
    # i, j, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i
                W[k * n + i][col] = 1
                col += 1
    # col is now 64
    
    # 2. Right Border: C[i, 4] = sum(j=0..3) A[i, j] B[j, 4]
    # i in 0..3, j in 0..3
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    # col is now 80
    
    # 3. Bottom Border: C[4, k] = sum(j=0..3) A[4, j] B[j, k]
    # k in 0..3, j in 0..3
    for k in range(4):
        for j in range(4):
            U[4 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 4][col] = 1
            col += 1
    # col is now 96
    
    # 4. Corner: C[4, 4] = A[4, 4] B[4, 4]
    U[4 * m + 4][col] = 1
    V[4 * p + 4][col] = 1
    W[4 * n + 4][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
