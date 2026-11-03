import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = 2, 5, 5
    rank = n * m * p  # 50
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # The task requires integer/half-integer entries.
    # The evaluator expects a dictionary with 'U', 'V', 'W' keys.
    # The values should be lists of lists or arrays.
    # Based on the seed program, it returns a dict of lists.
    
    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
