import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    # Start time to track elapsed time
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists to match the expected output format
    # The task specifies: {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # Entries should be integer/half-integer.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Return the decomposition
    return {"U": U, "V": V, "W": W}
