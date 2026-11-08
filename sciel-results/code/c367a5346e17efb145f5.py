"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for matrix multiplication of dimensions 2x3 by 3x5.
    This is a valid decomposition of rank 30.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # The tensor is sum over i, j, k of E_{ij}^A \otimes E_{jk}^B \otimes E_{ik}^C
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
