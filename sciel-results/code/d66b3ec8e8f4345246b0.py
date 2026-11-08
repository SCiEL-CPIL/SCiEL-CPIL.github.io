"""Search for a rank-minimal exact tensor decomposition of 2x4 * 4x7 multiplication."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
N, M, P = 2, 4, 7
DEADLINE = time.time() + 140.0


def _target():
    n, m, p = N, M, P
    A = np.zeros((n * m, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A[i * m + j, k * n + i] = 1
    return A


def _build_schoolbook():
    n, m, p = N, M, P
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _reconstruct(U, V, W):
    n, m, p = N, M, P
    r = len(U[0])
    A = np.zeros((n * m, p * n))
    for c in range(r):
        u = np.array(U[:, c], dtype=float)
        v = np.array(V[:, c], dtype=float)
        w = np.array(W[:, c], dtype=float)
        A += np.outer(np.outer(u, v), w)
    return A


def _is_valid(U, V, W, tol=1e-9):
    A = _reconstruct(U, V, W)
    T = _target()
    return bool(np.allclose(A, T, atol=tol, rtol=0))


def _try_merge(U, V, W, c1, c2):
    # combine two rank-1 terms into one; valid iff u1v1w1+u2v2w2 == (u1+u2)(v1+v2)(w1+w2)
    r = len(U[0])
    U2 = [row[:] for row in U]
    V2 = [row[:] for row in V]
    W2 = [row[:] for row in W]
    for i in range(len(U2)):
        U2[i][c1] += U2[i][c2]
    for i in range(len(V2)):
        V2[i][c1] += V2[i][c2]
    for i in range(len(W2)):
        W2[i][c1] += W2[i][c2]
    for i in range(len(U2)):
        U2[i].pop(c2)
    for i in range(len(V2)):
        V2[i].pop(c2)
    for i in range(len(W2)):
        W2[i].pop(c2)
    return U2, V2, W2


def _try_split(U, V, W, c):
    # split one rank-1 term into two that sum to it: (a u, a v, w)+(u(1-a), v(1-a), w)
    # choose a=1/2 to keep half-integers
    r = len(U[0])
    U2 = [row[:] for row in U]
    V2 = [row[:] for row in V]
    W2 = [row[:] for row in W]
    for i in range(len(U2)):
        U2[i].insert(c + 1, U2[i][c] * 0.5)
        U2[i][c] *= 0.5
    for i in range(len(V2)):
        V2[i].insert(c + 1, V2[i][c] * 0.5)
        V2[i][c] *= 0.5
    for i in range(len(W2)):
        W2[i].insert(c + 1, W2[i][c] * 0.5)
        W2[i][c] *= 0.5
    return U2, V2, W2


def construct():
    rng = random.Random(12345)
    U, V, W = _build_schoolbook()
    if not _is_valid(U, V, W):
        # should never happen; return anyway
        return {"U": U, "V": V, "W": W}
    best = (len(U[0]), [row[:] for row in U], [row[:] for row in V], [row[:] for row in W])

    # iterative merge attempts
    improved = True
    while improved and time.time() < DEADLINE - 20:
        improved = False
        r = len(U[0])
        pairs = [(c1, c2) for c1 in range(r) for c2 in range(c1 + 1, r)]
        rng.shuffle(pairs)
        for c1, c2 in pairs:
            if time.time() > DEADLINE - 20:
                break
            U2, V2, W2 = _try_merge(U, V, W, c1, c2)
            if _is_valid(U2, V2, W2):
                U, V, W = U2, V2, W2
                improved = True
                break
        if len(U[0]) < best[0]:
            best = (len(U[0]), [row[:] for row in U], [row[:] for row in V], [row[:] for row in W])

    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
