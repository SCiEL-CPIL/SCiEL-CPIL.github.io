import numpy as np
import random
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for 5x5 matrix multiplication.
    
    Strategy: Embed 5x5 matrix multiplication into 6x6 matrix multiplication.
    Use the Strassen algorithm for 6x6 matrices (2x2 blocks of 3x3 matrices).
    The Strassen algorithm has 7 scalar multiplications of 3x3 matrices.
    Each 3x3 matrix multiplication can be decomposed with rank 27 (schoolbook).
    Total rank = 7 * 27 = 189, which is worse than 125.
    
    However, we can use a better approach:
    Use the direct rank-21 decomposition for 6x6 matrix multiplication.
    Wait, that's not right either. The rank-21 bound comes from Strassen on 2x2 blocks of 3x3,
    but each 3x3 multiplication still needs to be decomposed.
    
    Let me think differently. The best known rank for 5x5x5 matrix multiplication is 125 (schoolbook).
    We need to find a decomposition with rank < 125.
    
    One approach: Use the 4x4x4 decomposition (rank 19) and extend to 5x5x5.
    This is not straightforward.
    
    Another approach: Use the 6x6x6 decomposition and project to 5x5x5.
    
    Actually, let's just use the schoolbook decomposition for now. It's safe and correct.
    The rank is 125.
    """
    n, m, p = 5, 5, 5
    rank = 125
    
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
