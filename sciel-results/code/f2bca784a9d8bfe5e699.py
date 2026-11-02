"""Optimized tensor decomposition for 2x2 @ 2x11 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    def verify(U, V, W, tol=1e-10):
        n, m, p = DIMENSIONS
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    U, V, W = make_schoolbook()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": [[0] for _ in range(4)], "V": [[0] for _ in range(22)], "W": [[0] for _ in range(22)]}
# EVOLVE-BLOCK-END
