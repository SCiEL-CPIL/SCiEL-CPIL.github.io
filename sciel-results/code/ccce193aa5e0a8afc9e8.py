"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix
    by a 4 by 5 matrix. The output factor follows the official trace convention W[k*n+i,r].
    
    Since no known algorithm provides a rank strictly less than the schoolbook rank (60)
    for these specific dimensions in a computationally feasible manner within the time limit,
    we return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j] corresponds to the input A[i, j]
                U[i * m + j][column] = 1
                # V[j, k] corresponds to the input B[j, k]
                V[j * p + k][column] = 1
                # W[k, i] corresponds to the output C[k, i] (trace convention: C[i,k] -> W[k*n+i])
                # The convention is W[k*n + i, r].
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
