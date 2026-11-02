"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for 3x4x5 matrix multiplication.
    Rank = 3*4*5 = 60.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input indices (i, j) to column
                U[i * m + j][column] = 1
                # V maps input indices (j, k) to column
                V[j * p + k][column] = 1
                # W maps output indices (k, i) to column
                # Note: W[k*n+i] corresponds to the coefficient for C[i,k]
                # The tensor component is C[i,k] += A[i,j] * B[j,k]
                # So W index is (k, i) in the "output" space of dimension p*n
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
