import numpy as np
import time

def construct():
    n, m, p = 2, 5, 5
    rank = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Initialize matrices with zeros
    # Using lists as per original, but ensuring they are exact integers
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time
                if time.time() > deadline:
                    break
                
                # Index mapping:
                # U row: i * m + j
                # V row: j * p + k
                # W row: k * n + i (following W[k*n+i, r] convention)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
