"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best = {'U': U_school.tolist(), 'V': V_school.tolist(), 'W': W_school.tolist(), 'rank': rank_school}
    
    # No sub-105 rank found quickly, return schoolbook
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
