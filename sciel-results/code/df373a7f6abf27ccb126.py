import numpy as np

def construct():
    n, m, p = 3, 4, 7
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Iterate with middle index j as the outer loop to build a structurally
    # different assembly of the same exact tensor decomposition.
    col = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
