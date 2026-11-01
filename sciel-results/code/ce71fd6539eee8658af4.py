"""Matrix multiplication 2x2 x 2x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a (2,2) matrix by a (2,8) matrix.
    
    Index conventions:
    - U has shape (n*m, rank) = (4, rank). Row a = i*m + j = i*2 + j.
    - V has shape (m*p, rank) = (16, rank). Row b = j*p + k = j*8 + k.
    - W has shape (p*n, rank) = (16, rank). Row c = k*n + i = k*2 + i.
    
    The tensor identity requires:
    sum_r U[a,r] * V[b,r] * W[c,r] = 1 if (i,j,k) are consistent, else 0.
    
    For C = A*B where A is 2x2, B is 2x8, C is 2x8:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We use the schoolbook decomposition with rank 32 (one rank-1 term per (i,j,k) triple).
    """
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    rank = n * m * p  # 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 16 x 32
    W = [[0] * rank for _ in range(p * n)]  # 16 x 32
    
    # For each (i, j, k), create one rank-1 term
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row: a = i*m + j
                a = i * m + j
                # V row: b = j*p + k
                b = j * p + k
                # W row: c = k*n + i
                c = k * n + i
                
                U[a][r] = 1
                V[b][r] = 1
                W[c][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
