"""Rewritten program: search for rank-minimal exact tensor decomposition of [2,5,6] matmul."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
DEADLINE = 120.0
START = time.time()

def _time_left():
    return DEADLINE - (time.time() - START) - 5.0

def _verify(U, V, W, n, m, p, tol=1e-9):
    r = U.shape[1]
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    prod = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.max(np.abs(prod - 1.0)) < tol

def _try_rank(r, n, m, p, max_iters=300, seed=0, tol=1e-9):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n * m, r))
    V = rng.standard_normal((m * p, r))
    W = rng.standard_normal((p * n, r))
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        for i in range(n):
            for j in range(m):
                A = (Vr[j] * Wr[:, i]).T  # (r, p)
                sol, *_ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                Ur[i, j] = sol
        for j in range(m):
            for k in range(p):
                A = (Ur[:, j] * Wr[k]).T  # (r, n)
                sol, *_ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                Vr[j, k] = sol
        for k in range(p):
            for i in range(n):
                A = (Ur[i] * Vr[:, k]).T  # (r, m)
                sol, *_ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                Wr[k, i] = sol
        if it % 20 == 0:
            prod = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
            if np.max(np.abs(prod - 1.0)) < tol:
                return U, V, W
    return None

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

def _to_list(arr):
    out = []
    for row in arr:
        rrow = []
        for x in row:
            h = round(x * 2)
            if abs(h - x * 2) < 1e-7:
                rrow.append(int(h) / 2.0)
            else:
                rrow.append(float(x))
        out.append(rrow)
    return out

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    for r in range(8, 61):
        if _time_left() < 10.0:
            break
        for attempt in range(20):
            if _time_left() < 5.0:
                break
            res = _try_rank(r, n, m, p, max_iters=300, seed=attempt * 1000 + r, tol=1e-9)
            if res is not None:
                U, V, W = res
                Ur = np.round(U * 2) / 2
                Vr = np.round(V * 2) / 2
                Wr = np.round(W * 2) / 2
                if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
                    best = {"U": _to_list(Ur), "V": _to_list(Vr), "W": _to_list(Wr)}
                    return best
    return best
