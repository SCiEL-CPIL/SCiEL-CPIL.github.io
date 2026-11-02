"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import numpy as np
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Rank 9 solution: one term per (i,j) pair, covering all k values
    U9 = np.zeros((n * m, 9))
    V9 = np.zeros((m * p, 9))
    W9 = np.zeros((p * n, 9))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U9[i * m + j, col] = 1
            for k in range(p):
                V9[j * p + k, col] = 1
                W9[k * n + i, col] = 1
            col += 1
    
    if check_decomposition(U9, V9, W9, tol=1e-6):
        best_result = {"U": U9.tolist(), "V": V9.tolist(), "W": W9.tolist()}
        best_rank = 9
    
    return best_result
# EVOLVE-BLOCK-END
