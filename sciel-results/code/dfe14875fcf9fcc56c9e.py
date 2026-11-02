import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Construct the schoolbook decomposition
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                U[a][column] = 1
                V[b][column] = 1
                W[c][column] = 1
                
                column += 1
                
    return {"U": U, "V": V, "W": W}
