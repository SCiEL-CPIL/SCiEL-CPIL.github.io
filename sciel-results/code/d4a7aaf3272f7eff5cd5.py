"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """
    Return a valid schoolbook decomposition as (U, V, W) lists of lists of floats.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1 for all i,j,k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries (as floats).
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    
    # Generate schoolbook decomposition (rank 80)
    # This is a valid, though not minimal, decomposition.
    U, V, W = _schoolbook(n, m, p)
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Ensure all values are standard Python floats for JSON serialization
    # The lists already contain floats, but we convert to be safe.
    result = {
        "U": U,
        "V": V,
        "W": W
    }
    
    return result
# EVOLVE-BLOCK-END
