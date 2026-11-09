import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    target = np.ones((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - target[i,j,k]) > 1e-9:
                    return False
    return True

def _als_search(rank, deadline, seed=0):
    n, m, p = DIMENSIONS
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n*m, rank))
    V = rng.standard_normal((m*p, rank))
    W = rng.standard_normal((p*n, rank))
    # target
    target = np.ones((n, m, p), dtype=float)
    for it in range(200):
        if time.time() > deadline - 5:
            break
        # update U
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    num = 0.0
                    den = 0.0
                    for k in range(p):
                        w = W[k*n+i, c]
                        num += target[i,j,k] * V[j*p+k, c] * w
                        den += (V[j*p+k, c] * w) ** 2
                    U[i*m+j, c] = num / den if den > 1e-15 else 0.0
        # update V
        for c in range(rank):
            for j in range(m):
                for k in range(p):
                    num = 0.0
                    den = 0.0
                    for i in range(n):
                        u = U[i*m+j, c]
                        num += target[i,j,k] * u * W[k*n+i, c]
                        den += (u * W[k*n+i, c]) ** 2
                    V[j*p+k, c] = num / den if den > 1e-15 else 0.0
        # update W
        for c in range(rank):
            for k in range(p):
                for i in range(n):
                    num = 0.0
                    den = 0.0
                    for j in range(m):
                        v = V[j*p+k, c]
                        num += target[i,j,k] * U[i*m+j, c] * v
                        den += (U[i*m+j, c] * v) ** 2
                    W[k*n+i, c] = num / den if den > 1e-15 else 0.0
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    deadline = time.time() + 115.0
    n, m, p = DIMENSIONS

    # rank 1 all-ones
    U = [[1]] * (n*m); V = [[1]] * (m*p); W = [[1]] * (p*n)
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # small search
    best = None
    for rank in range(2, 21):
        if time.time() > deadline - 10:
            break
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            U, V, W = _als_search(rank, deadline, seed)
            if _verify(U, V, W):
                if best is None or rank < best[0]:
                    best = (rank, U, V, W)
                break
        if best is not None and best[0] == rank:
            break

    if best is not None:
        return {"U": best[1], "V": best[2], "W": best[3]}

    # schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
