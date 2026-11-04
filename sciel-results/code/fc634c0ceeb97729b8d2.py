"""Matrix multiplication 3x4x8 - rank 96 via schoolbook decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    rank = n * m * p  # 96
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify the decomposition on a sample of entries
    for _ in range(200):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        a = i * m + j
        b = j * p + k
        c = k * n + i
        val = float(np.sum(U[a, :] * V[b, :] * W[c, :]))
        if abs(val - 1.0) > 1e-10:
            raise AssertionError(f"Verification failed at ({i},{j},{k}): got {val}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
