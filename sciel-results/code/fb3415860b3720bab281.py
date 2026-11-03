"""Matrix multiplication 2x3x9 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    # nm=6, mp=27, pn=18

    # We use the schoolbook decomposition which is guaranteed to be correct.
    # Rank = n*m*p = 54.
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

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
# EVOLVE-BLOCK-END
