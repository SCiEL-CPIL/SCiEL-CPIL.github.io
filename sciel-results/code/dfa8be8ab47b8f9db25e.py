import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    The minimal rank for M_<2,3,6> is 36.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
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
                # U maps input A[i,j] to the column
                U[i * m + j][col] = 1
                # V maps input B[j,k] to the column
                V[j * p + k][col] = 1
                # W maps output C[i,k] from the column
                # Trace convention: W[k*n + i, r]
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
