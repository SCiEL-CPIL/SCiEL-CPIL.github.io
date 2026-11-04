"""
Rank-27 exact tensor decomposition for 3x3 matrix multiplication.
Uses the schoolbook algorithm.
"""

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _check_half_ints(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for v in row:
                if abs(v * 2 - round(v * 2)) > tol:
                    return False
    return True

def construct():
    """
    Return a rank-27 decomposition of the 3x3 matrix multiplication tensor.
    This is the schoolbook algorithm.
    """
    n, m, p = DIMENSIONS  # 3, 3, 3
    
    rank = n * m * p  # 27
    U = [[0] * rank for _ in range(n * m)]  # 9 x 27
    V = [[0] * rank for _ in range(m * p)]  # 9 x 27
    W = [[0] * rank for _ in range(p * n)]  # 9 x 27
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verification
    if not _verify_decomposition(U, V, W):
        raise ValueError("Decomposition failed verification")
    if not _check_half_ints(U, V, W):
        raise ValueError("Entries are not half-integers")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
