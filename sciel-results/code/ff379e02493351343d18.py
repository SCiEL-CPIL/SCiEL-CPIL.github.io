"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook decomposition (rank 105).
    2. Use Strassen-like reductions on 2x2 sub-blocks where possible.
    3. Try to find a low-rank decomposition using numerical optimization or known algorithms.
    
    Given the complexity, we'll implement a hybrid approach:
    - Use Strassen for 2x2x2 blocks to reduce rank.
    - For the remaining parts, use schoolbook.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook decomposition
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
    
    # Try Strassen reduction for 2x2 sub-blocks
    # For a 2x2x2 block, Strassen reduces rank from 8 to 7
    # We can apply this to overlapping 2x2 blocks in the matrix
    
    # Let's try a different approach: use the known minimal rank for small matrices
    # For 3x5x7, the exact minimal rank is unknown, but we can try to find a good decomposition
    
    # Use numerical approach to find a lower rank decomposition
    # This is computationally expensive, so we'll try a few iterations
    
    # For now, return the schoolbook decomposition as a baseline
    # A more advanced approach would use tensor decomposition algorithms
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
