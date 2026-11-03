"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
"""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions n x m by m x p.
    
    The decomposition is:
    C = A * B
    where A is n x m, B is m x p, C is n x p.
    
    The tensor T[i, j, k] corresponds to the coefficient of A[i, j] * B[j, k] in C[i, k].
    For standard matrix multiplication, T[i, j, k] = 1 if the indices are valid, 0 otherwise.
    Actually, the tensor is defined such that:
    C[i, k] = sum_j A[i, j] B[j, k]
    
    The rank-r decomposition is given by:
    sum_{c=1}^r U[i, j, c] V[j, k, c] W[k, i, c] = T[i, j, k]
    
    Wait, the interface specifies:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    And the trace convention is W[k*n+i, r].
    The condition for exact decomposition is:
    sum_{c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid i, j, k.
    
    The schoolbook algorithm uses rank = n*m*p.
    For each (i, j, k), we assign a unique column c.
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    All other entries are 0.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists to match the expected output format strictly, 
    # though numpy is faster to build. The prompt says "integer/half-integer entries".
    # We will use integers (0 or 1).
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
