import time
import random
import numpy as np

TIME_LIMIT = 120.0
START_TIME = time.time()

def construct():
    global START_TIME, TIME_LIMIT
    n, m, p = 2, 4, 7
    
    # Target dimensions: U (n*m x R), V (m*p x R), W (p*n x R)
    # Index conventions:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # Trivial Rank = 56
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    construct()
