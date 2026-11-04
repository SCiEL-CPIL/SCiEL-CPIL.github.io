"""Matrix multiplication 6x6x6 - Schoolbook decomposition (rank 216)."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Standard schoolbook matrix multiplication decomposition for 6x6 matrices.
    Rank = 6 * 6 * 6 = 216.
    """
    start_time = time.time()
    deadline = start_time + 100

    n = m = p = 6
    rank = 216

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                a_idx = i * m + j
                # B index: j*p + k
                b_idx = j * p + k
                # C index: k*n + i (following W[k*n+i, r] convention)
                c_idx = k * n + i
                
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
