import time
import sys

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for matrix multiplication 3x4 by 4x8.
    Currently uses the schoolbook decomposition of rank 96.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
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
                
    return {"U": U, "V": V, "W": W}
