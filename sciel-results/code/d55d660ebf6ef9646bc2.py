import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                a_idx = i * m + j
                # B index: j*p + k
                b_idx = j * p + k
                # C index: k*n + i
                c_idx = k * n + i
                
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
