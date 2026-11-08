import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = 110.0 # 10 seconds safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    U = [[0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0 for _ in range(rank)] for _ in range(p * n)]
    
    # Fill the schoolbook decomposition
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
                # Time check (unlikely to trigger for 30 iters, but required)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
