"""
Matrix multiplication 3,4,6 rank-minimal exact tensor decomposition.
Uses the schoolbook decomposition of rank n*m*p.
"""
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input A[i,j] to the rank-th component
                U[i * m + j][col] = 1
                # V maps input B[j,k] to the rank-th component
                V[j * p + k][col] = 1
                # W maps the product to output C[i,k]
                # Note: The convention is W[k*n + i, r]
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
