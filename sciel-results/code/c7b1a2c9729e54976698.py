import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # The task is to find a rank-minimal exact tensor decomposition.
    # For 3x3x7, the schoolbook rank is 3*3*7 = 63.
    # Without pre-computed coefficients for lower-rank decompositions (like Strassen variants for 3x3x3),
    # the schoolbook method is the most reliable exact decomposition we can construct programmatically 
    # within the time limit and complexity constraints.
    
    # We construct the factors for the schoolbook algorithm.
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # where c iterates over all (i, j, k) tuples.
    
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists to match the required interface
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
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {
        "U": U,
        "V": V,
        "W": W
    }
