"""
Rank-minimal exact tensor decomposition for 3x3 by 3x7 matrix multiplication.

Uses schoolbook decomposition (rank 63) as a correct baseline.
"""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns {'U', 'V', 'W'} with integer/half-integer entries that exactly
    decompose the 3x3 * 3x7 matrix multiplication tensor.
    Uses schoolbook decomposition with rank 63.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    rank = n * m * p  # 63
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
