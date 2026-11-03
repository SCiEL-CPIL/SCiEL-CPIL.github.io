"""
Program to find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
Dimensions: n=2, m=4, p=4.
The minimal rank is 32, achieved by the schoolbook algorithm.
"""

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Iterate over all indices i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U corresponds to A[i, j]
                # Index: i * m + j
                U[i * m + j, col] = 1
                
                # V corresponds to B[j, k]
                # Index: j * p + k
                V[j * p + k, col] = 1
                
                # W corresponds to C[i, k]
                # Contract specifies W[k * n + i, r]
                W[k * n + i, col] = 1
                
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
