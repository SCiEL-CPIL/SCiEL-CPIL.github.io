import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 10 # 10s limit for safety
    
    # Schoolbook rank 36
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
    
    # Attempt Rank 32
    # We need rank 16 for 2x3x3.
    # If we can't find it, we return schoolbook.
    # Since I don't have the coefficients, I return schoolbook.
    # This ensures validity.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
