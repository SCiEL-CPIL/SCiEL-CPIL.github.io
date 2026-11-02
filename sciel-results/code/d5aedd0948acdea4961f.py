"""Matrix multiplication tensor decomposition for 3x3 x 3x6."""
import time
import math
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Returns dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Schoolbook decomposition: rank = 54
    rank = n * m * p
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
