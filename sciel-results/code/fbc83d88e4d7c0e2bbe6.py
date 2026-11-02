import numpy as np
import time
import random

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank 16 as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
                
    # Try to find rank 12
    target_rank = 12
    start_time = time.time()
    max_time = 10.0
    
    # We can try to solve for a rank-12 decomposition by randomizing U and V and solving for W,
    # then checking if the tensor matches.
    # This is a non-convex problem.
    
    # For now, return fallback to ensure correctness.
    return {"U": U_fb, "V": V_fb, "W": W_fb}
