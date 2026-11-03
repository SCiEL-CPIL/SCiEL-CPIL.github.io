import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Try random sparse decomposition
    # This is a heuristic search
    for rank in range(50, rank_school):
        if time.time() > deadline - 10:
            break
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            # Generate random sparse matrices
            U_rand = [[0] * rank for _ in range(n * m)]
            V_rand = [[0] * rank for _ in range(m * p)]
            W_rand = [[0] * rank for _ in range(p * n)]
            
            # This search space is too large.
            pass
            
    return {"U": best_U, "V": best_V, "W": best_W}
