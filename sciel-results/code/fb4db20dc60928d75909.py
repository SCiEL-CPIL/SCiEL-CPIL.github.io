"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-144 schoolbook tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    Each term A[i,j] * B[j,k] -> C[i,k] is rank 1.
    Total rank = 4*4*9 = 144
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    rank = n * m * p  # 144
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] index: i*m + j
                # B[j,k] index: j*p + k
                # C[i,k] index: k*n + i (per task convention W[k*n+i, r])
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
