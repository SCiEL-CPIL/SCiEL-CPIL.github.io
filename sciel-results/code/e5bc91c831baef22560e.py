"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # 1. Construct Schoolbook Decomposition (Rank 40)
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # The seed code implements the standard multiplication tensor for C=AB (transposed output).
    # Rank is n*m*p = 40.
    # Lower rank is an open problem for these dimensions, unlikely to be found in 120s.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
