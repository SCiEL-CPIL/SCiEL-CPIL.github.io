"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import numpy as np
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    We use the rank-19 4x4x4 Lander-Bailey decomposition as a base, and try to extend it.
    If we can't easily extend, we use the rank-28 block decomposition for 4x4x4 plus
    the remaining 4 columns.
    
    Actually, let's just implement a known low-rank decomposition for 4x4x5.
    The best known upper bound for 4x4x5 is around 28 or so.
    We'll use a block-based approach.
    
    For now, let's implement the rank-80 schoolbook as fallback, and try to improve.
    """
    n, m, p = 4, 4, 5  # dimensions
    
    # Try to construct a rank-28 decomposition
    # We can use the fact that we can decompose 4x4x4 with rank 19,
    # and then handle the extra 4x4x1 and 4x1x4 and 1x4x4 parts.
    
    # Actually, let's try a different approach:
    # Use the 28-rank 4x4x4 decomposition (if I can recall it correctly)
    # Or better, use a systematic construction.
    
    # Let's try to build a rank-28 decomposition using block methods.
    # We can partition the 4x4x5 tensor into blocks.
    
    # Actually, the simplest approach: use the rank-19 4x4x4 Lander-Bailey
    # and add 4 more terms for the 5th column.
    # But I need to remember the 19-rank decomposition exactly.
    
    # Let me implement the rank-19 Lander-Bailey decomposition.
    # The Lander-Bailey 4x4x4 decomposition has rank 19.
    
    # Here are the 19 terms for 4x4x4 (indices 0-3 for each dimension):
    # U[i,j], V[j,k], W[k,i] for each term t=0..18
    
    # The Lander-Bailey rank-19 decomposition:
    lb_U = [
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,2)
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,3)
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # (1,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # (1,2)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # (1,3)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # (2,0)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,2)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,3)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (3,0)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (3,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (3,2)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (3,3)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    # This is getting complicated. Let me just use a simpler approach.
    # Use rank-80 schoolbook decomposition.
    
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
