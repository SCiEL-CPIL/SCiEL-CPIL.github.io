import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    # The task is to decompose the tensor for multiplying a 2x2 matrix by a 2x9 matrix.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # This is a direct sum of 9 independent 2x2x1 matrix-vector products.
    # The rank of a 2x2x1 multiplication is 4.
    # Total rank = 9 * 4 = 36.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
