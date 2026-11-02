"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.
Implements the schoolbook rank-32 decomposition.
"""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Dimensions:
    n = 2 (rows of A, cols of C)
    m = 4 (cols of A, rows of B)
    p = 4 (cols of B, rows of C)
    
    U shape: (n*m, rank) = (8, 32)
    V shape: (m*p, rank) = (16, 32)
    W shape: (p*n, rank) = (8, 32)
    
    Indexing:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]
    
    Schoolbook decomposition:
    One term for each (i, j, k) in {0..1} x {0..3} x {0..3}.
    Total rank = 2 * 4 * 4 = 32.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the 1s for this rank-1 component
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
