import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition: Rank = n * m * p
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    k = 0
    for i in range(n):
        for j in range(p):
            for l in range(m):
                u_idx = i * m + l
                v_idx = l * p + j
                w_idx = j * n + i
                
                U[u_idx][k] = 1.0
                V[v_idx][k] = 1.0
                W[w_idx][k] = 1.0
                
                k += 1
                
    return {"U": U, "V": V, "W": W}
