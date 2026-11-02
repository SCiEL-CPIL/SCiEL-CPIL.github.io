import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # Return what we have so far? No, it must be a complete decomposition.
                    # If we run out of time, we should just return the full schoolbook.
                    # But the loop is fast. 125 iterations.
                    pass
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
