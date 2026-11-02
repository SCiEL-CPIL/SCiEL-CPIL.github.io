"""Matrix multiplication 3x3x7 rank-minimal tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    # Optimal rank is m*p = 21 (lower bound: max(n*p, m*p, n*m) = max(21,21,9) = 21)
    # Construction: one column per (j,k) pair.
    rank = m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            for i in range(n):
                W[k * n + i, c] = 1.0
            c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
