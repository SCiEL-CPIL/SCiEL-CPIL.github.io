import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use naive decomposition: rank = n*m*p = 4*4*8 = 128
    # This is guaranteed correct
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For naive matrix multiplication:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # In tensor form: T[i,k,j] = A[i,k] * B[k,j]
    # The tensor decomposition has rank n*m*p where each term is:
    # U[i,k, r] = 1 if r corresponds to (i,k,j), else 0
    # V[k,j, r] = 1 if r corresponds to (i,k,j), else 0  
    # W[j,i, r] = 1 if r corresponds to (i,k,j), else 0
    # Note: W uses column-major indexing: W[j*n + i, r]
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = i * m * p + k * p + j
                U[i * m + k][r] = 1
                V[k * p + j][r] = 1
                # W index: j*n + i (column-major)
                W[j * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
