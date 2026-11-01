import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Texp = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                Texp[a, b, c] = 1.0
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Block decomposition: for each j in 0..m-1, use n ranks
    # Total rank = m * n
    rank = m * n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for j in range(m):
        for i in range(n):
            r = j * n + i
            U[i * m + j][r] = 1
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1

    result = {"U": U, "V": V, "W": W}

    if _verify(U, V, W, n, m, p):
        return result

    return _schoolbook(n, m, p)
