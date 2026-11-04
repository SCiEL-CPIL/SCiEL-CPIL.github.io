import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    dims = DIMENSIONS
    n, m, p = dims
    r = n * m * p
    U = np.zeros((n * m, r), dtype=int)
    V = np.zeros((m * p, r), dtype=int)
    W = np.zeros((p * n, r), dtype=int)
    for idx in range(r):
        i = idx % n
        j = (idx // n) % m
        k = (idx // (n * m)) % p
        U[i * m + j, idx] = 1
        V[j * p + k, idx] = 1
        W[k * n + i, idx] = 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
