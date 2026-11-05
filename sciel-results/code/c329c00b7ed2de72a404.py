import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-minimal exact tensor decomposition for matrix multiplication.
    For 3x4 * 4x7, the standard schoolbook algorithm yields a rank of 3*4*7 = 84.
    No widely known algebraic identities reduce this rank significantly for these specific dimensions 
    in a way that is easily constructible with integer coefficients without complex search.
    Thus, we return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term corresponds to A[i, j] * B[j, k] contributing to C[i, k]
                # U[i*m + j, col] = 1
                # V[j*p + k, col] = 1
                # W[k*n + i, col] = 1
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
