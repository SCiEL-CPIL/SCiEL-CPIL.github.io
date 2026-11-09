import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin before 150s limit
    
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
                
                # Check time periodically
                if time.time() > deadline:
                    # Return partial or best found (here schoolbook is complete)
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
