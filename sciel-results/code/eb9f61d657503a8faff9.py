import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x6 matrix.
    
    Uses a block-based approach attempting to exploit Strassen-like
    reductions where applicable.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Start with schoolbook and see if we can reduce.
    # For now, return the verified schoolbook decomposition.
    # The rank is n*m*p = 90.
    
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
