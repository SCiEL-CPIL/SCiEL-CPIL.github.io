"""Matrix multiplication 2,3,7 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Return U, V, W factor matrices for 2x3 * 3x7 matrix multiplication.
    
    Uses a block-wise decomposition: split the 2x3 matrix into two row-vectors,
    each multiplied by the 3x7 matrix. For each row-vector (1x3) times (3x7),
    use the schoolbook dot-product approach: rank 3*7=21 per row, but since
    both rows share the same B matrix, we can potentially save rank.
    
    Actually, the cleanest valid decomposition is schoolbook: rank = 2*3*7 = 42.
    
    Let me verify: the tensor is sum over i,j,k of A[i,j]*B[j,k]*C[k,i].
    Schoolbook: one rank-1 term for each (i,j,k) triple. Rank = n*m*p = 42.
    
    This is always valid. Let's return it.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
