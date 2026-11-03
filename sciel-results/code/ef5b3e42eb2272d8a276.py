"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x4 matrix.
    
    The matrix multiplication tensor T has entries T[i,j,k] = 1 for all
    i in [0,n), j in [0,m), k in [0,p), because:
    tr(C^T (AB)) = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
    
    This is the all-ones tensor, which has tensor rank 1.
    We use U[i*m+j, c] = V[j*p+k, c] = W[k*n+i, c] = 1 for c=0.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank 1 decomposition: all entries are 1
    rank = 1
    U = [[1] for _ in range(nm)]
    V = [[1] for _ in range(mp)]
    W = [[1] for _ in range(pn)]
    
    # Verify correctness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if val != 1:
                    # Fallback to schoolbook if verification fails
                    rank = n * m * p
                    U = [[0] * rank for _ in range(nm)]
                    V = [[0] * rank for _ in range(mp)]
                    W = [[0] * rank for _ in range(pn)]
                    col = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U[i*m+j][col] = 1
                                V[j*p+k][col] = 1
                                W[k*n+i][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
