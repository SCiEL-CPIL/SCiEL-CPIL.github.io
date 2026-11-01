import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145

def _time_exceeded():
    return time.time() - _start_time > TIME_LIMIT

def construct():
    n, m, p = 2, 2, 15
    rank_school = n * m * p  # 60
    
    # Schoolbook decomposition
    # U shape: (n*m, rank) = (4, 60)
    # V shape: (m*p, rank) = (30, 60)
    # W shape: (p*n, rank) = (30, 60)
    
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                # B index: j*p + k
                # C index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
