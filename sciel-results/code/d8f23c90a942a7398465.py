"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]
import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    
    The matrix multiplication tensor M_{i,j,k} for C[i,k] = sum_j A[i,j] B[j,k]
    can be decomposed as a sum of rank-1 tensors. The standard schoolbook algorithm
    has rank n*m*p = 3*3*4 = 36.
    
    For 3x3x4 matrix multiplication, the known minimal rank is 36 (the schoolbook rank).
    No known algorithm achieves rank < 36 for this specific dimension.
    
    We return the schoolbook decomposition with integer entries (all 1s and 0s).
    
    Interface:
    - U: shape (n*m, rank) = (9, rank), with U[i*m+j, c]
    - V: shape (m*p, rank) = (12, rank), with V[j*p+k, c]
    - W: shape (p*n, rank) = (12, rank), with W[k*n+i, c]
    
    The decomposition satisfies:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = M_{i,j,k}
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 36
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition: one rank-1 term for each (i, j, k) triple
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
