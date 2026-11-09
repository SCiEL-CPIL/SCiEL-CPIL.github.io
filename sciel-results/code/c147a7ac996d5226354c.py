import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook rank 54 decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Verify correctness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): {val}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
