import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # ... (code from above) ...
    
    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook
        n, m, p = 3, 5, 6
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

    U, V, W = get_356()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
