"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave margin for 150s limit
    
    # Schoolbook decomposition: rank = n*m*p = 105
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Check if we have time to try a better construction
    # For 3x5x7, a block decomposition using a known lower-rank 3x3x3 tensor
    # could potentially reduce the rank, but finding/verifying such a tensor
    # with exact integer/half-integer entries within the time limit is risky.
    # The schoolbook decomposition is guaranteed correct.
    
    return best_result
# EVOLVE-BLOCK-END
