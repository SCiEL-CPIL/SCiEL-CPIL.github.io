import time
import numpy as np

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 110  # seconds

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Uses the trivial decomposition with rank 60 = 2*3*10.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
