"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    # For 2x3 x 3x8, we can split into 2x3 x 3x2 blocks (4 blocks) or other strategies.
    # Known result: rank(2x3 x 3x8) <= 44 using Strassen on 2x2 blocks? 
    # Actually, let's just return the schoolbook rank 48 for now as a safe baseline, 
    # or try to find a known lower bound.
    # However, the task asks for rank-minimal. 
    # Schoolbook rank = 2*3*8 = 48.
    # If we can find a decomposition with rank < 48, it's better.
    # Let's try to construct a Strassen-like decomposition.
    # Split A (2x3) into A1 (2x1) and A2 (2x2)? No, Strassen works on square blocks.
    # We can pad to 3x3? Or use 2x2 blocks.
    # Let's just return the schoolbook decomposition to ensure validity, 
    # then evolve to find better.
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
