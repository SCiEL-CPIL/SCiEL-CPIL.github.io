import time
import numpy as np

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy: Use schoolbook decomposition with rank 54, which is guaranteed correct.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 120
    
    # Schoolbook decomposition: rank = n*m*p = 54
    rank = n * m * p
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
    
    return {"U": U, "V": V, "W": W}
