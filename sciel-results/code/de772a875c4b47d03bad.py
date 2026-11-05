"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We aim for rank 25.
    # Use BCDL rank 19 for 3x3x3 core.
    # The remaining parts are 3x3x1, 3x1x3, 3x1x1.
    # 3x3x1 rank 3, 3x1x3 rank 3, 3x1x1 rank 1. Total 19+3+3+1 = 26.
    # Can we do better?
    # Let's try to construct it.
    
    # BCDL rank 19 decomposition for 3x3x3
    # U, V, W are 9x19
    U_bcdl = np.zeros((9, 19))
    V_bcdl = np.zeros((9, 19))
    W_bcdl = np.zeros((9, 19))
    
    # BCDL entries
    # This is a known decomposition.
    # We will use a placeholder for now and refine.
    # Actually, let's use the schoolbook for the 3x3x3 part (rank 27) and see if we can improve.
    # Rank 27 + 3 + 3 + 1 = 34.
    # Rank 19 + 3 + 3 + 1 = 26.
    
    # Let's implement the BCDL rank 19 decomposition.
    # The BCDL decomposition is complex.
    # We will use a simpler approach: Strassen rank 7 for 2x2x2, but we have 3x3.
    # Let's stick to the schoolbook for now and try to optimize.
    
    # Schoolbook rank 48.
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
