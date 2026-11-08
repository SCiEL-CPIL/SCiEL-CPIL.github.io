"""Matrix multiplication 4x5x5: rank-minimal exact tensor decomposition.

Strategy:
  * Guarantee a valid schoolbook rank-100 decomposition (numpy, exact).
  * Run a time-bounded exact search for a lower rank: fix random half-integer
    U, V of rank r, solve for W per (k,i) block, round to half-integers, and
    verify the full 3-tensor identity. Accept the lowest verified r.
  * Return the best verified decomposition before the wall-clock deadline.
"""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]
WALL_CLOCK = 150.0
MARGIN = 8.0


def _build_T():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-6):
    r = U.shape[1]
    S = np.zeros((U.shape[0], V.shape[0], W.shape[0]))
    for c in range(r):
        S += U[:, c][:, None, None] * V[:, c][None, :, None] * W[:, c][None, None, :]
    return bool(np.max(np.abs(S - T)) < tol)


def _round_half(X):
    return np.round(2.0 * X) / 2.0


def _schoolbook():
    n, m, p = DIMENSIONS
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


def _try_rank(r, T, rng, max_trials, deadline):
    n, m, p = DIMENSIONS
    vals = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5])
    for _ in range(max_trials):
        if time.time() > deadline - MARGIN:
            return None
        U = rng.choice(vals, size=(n * m, r))
        V = rng.choice(vals, size=(m * p, r))
        W = np.zeros((p * n, r))
        ok = True
        for k in range(p):
            for i in range(n):
                M = np.stack([U[i * m + j, :] * V[j * p + k, :] for j in range(m)], axis=0)
                b = np.ones(m)
                w, *_ = np.linalg.lstsq(M, b, rcond=None)
                if np.max(np.abs(M @ w - b)) > 1e-6:
                    ok = False
                    break
                W[k * n + i, :] = w
        if not ok:
            continue
        Wr = _round_half(W)
        if _verify(U, V, Wr, T):
            return U, V, Wr
    return None


def construct():
    start = time.time()
    deadline = start + WALL_CLOCK
    T = _build_T()
    n, m, p = DIMENSIONS

    U0, V0, W0 = _schoolbook()
    best = (U0, V0, W0) if _verify(U0, V0, W0, T) else (U0, V0, W0)

    rng = np.random.default_rng(20240521)
    found = None
    for r in range(25, n * m * p):
        if time.time() > deadline - MARGIN:
            break
        res = _try_rank(r, T, rng, max_trials=15, deadline=deadline)
        if res is not None:
            found = res
            break

    if found is not None:
        best = found

    U, V, W = best
    return {"U": U, "V": V, "W": W}
