"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The correct decomposition for matrix multiplication tensor
    # T[i,j,k] = 1 if indices are consistent, else 0.
    # In the flattened space, this is a sparse tensor.
    # The schoolbook algorithm provides a valid decomposition with rank n*m*p.
    # Each term (i,j,k) corresponds to a rank-1 tensor that is 1 only at the
    # consistent triple (i*m+j, j*p+k, k*n+i) and 0 elsewhere.
    
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
