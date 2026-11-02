"""Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import sys

def construct():
    """
    Returns a dict with keys 'U', 'V', 'W' representing the factor matrices
    of a rank-minimal (or close to minimal) exact tensor decomposition
    for multiplying a 2x3 matrix by a 3x6 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Since finding the true minimal rank for this non-square case is hard,
    we use the schoolbook decomposition which is guaranteed to be exact.
    Rank = 2*3*6 = 36.
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = n * m * p  # 36
    
    # Schoolbook decomposition
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
                # U[i*m + j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, column] = 1 (trace convention: W[k*n + i])
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
