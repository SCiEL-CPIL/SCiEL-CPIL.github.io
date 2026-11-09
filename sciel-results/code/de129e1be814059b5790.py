import time
import numpy as np

DIMENSIONS = [3, 3, 7]
_START = time.time()
_DEADLINE = 120.0  # wall-clock safety margin (evaluator kills at 150s)


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _build_T8():
    """Return 8*T as an int64 array of shape (n*m, m*p, p*n).

    T is the matrix-multiplication tensor: T[i*m+j, j*p+k, k*n+i] = 1.
    Multiplying by 8 lets us search for integer factor matrices that are
    2x the (integer/half-integer) factors the interface expects.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return 8 * T


def _schoolbook():
    """Rank n*m*p exact decomposition (always valid)."""
    n, m, p = DIMENSIONS
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


def _check(U, V, W, T8):
    """Exact check that integer factors reconstruct 8*T."""
    if not (np.all(np.isfinite(U)) and np.all(np.isfinite(V)) and np.all(np.isfinite(W))):
        return False
    if max(np.max(np.abs(U)), np.max(np.abs(V)), np.max(np.abs(W))) > 1e6:
        return False
    R = np.einsum('ar,br,cr->abc', U, V, W)
    return bool(np.array_equal(R, T8))


def _als(T8, r, rng, iters=25, init=None):
    """Alternating least-squares on the 3-way tensor, then round to integers."""
    n, m, p = DIMENSIONS
    dU, dV, dW = n * m, m * p, p * n
    if init is None:
        U = rng.integers(-4, 5, size=(dU, r)).astype(float)
        V = rng.integers(-4, 5, size=(dV, r)).astype(float)
        W = rng.integers(-4, 5, size=(dW, r)).astype(float)
    else:
        U, V, W = [np.asarray(x, dtype=float) for x in init]
    T8f = T8.astype(float)
    T8u = T8f.reshape(dU, dV, dW)
    N = dU * dV
    for _ in range(iters):
        if _remaining() < 6.0:
            break
        # solve for W
        designW = (U[:, None, :] * V[None, :, :]).reshape(N, r)
        rhsW = T8u.reshape(N, dW).T
        Wsol, *_ = np.linalg.lstsq(designW, rhsW, rcond=None)
        W = Wsol.T
        # solve for U
        designU = (V[:, None, :] * W[None, :, :]).reshape(dV * dW, r)
        rhsU = T8u.reshape(dU, dV * dW).T
        Usol, *_ = np.linalg.lstsq(designU, rhsU, rcond=None)
        U = Usol.T
        # solve for V
        designV = (U[:, None, :] * W[None, :, :]).reshape(dU * dW, r)
        rhsV = T8u.reshape(dU * dW, dV).T
        Vsol, *_ = np.linalg.lstsq(designV, rhsV, rcond=None)
        V = Vsol.T
    return (np.round(U).astype(np.int64),
            np.round(V).astype(np.int64),
            np.round(W).astype(np.int64))


def construct():
    n, m, p = DIMENSIONS
    T8 = _build_T8()
    best = _schoolbook()
    best_rank = n * m * p
    rng = np.random.default_rng(20240517)

    # Warm-start material from the schoolbook decomposition.
    sb = _schoolbook()
    U63 = np.array(sb["U"], dtype=float)
    V63 = np.array(sb["V"], dtype=float)
    W63 = np.array(sb["W"], dtype=float)

    # Try to find an exact decomposition of rank < 63 via randomized ALS.
    for r in range(best_rank - 1, 24, -1):
        if _remaining() < 10.0:
            break
        restarts = 30
        for it in range(restarts):
            if _remaining() < 10.0:
                break
            init = None
            if it % 5 == 0:
                # Warm-start from a random projection of the schoolbook factors.
                M = rng.normal(size=(best_rank, r))
                init = (U63 @ M, V63 @ M, W63 @ M)
            try:
                Ur, Vr, Wr = _als(T8, r, rng, iters=25, init=init)
            except Exception:
                continue
            if _check(Ur, Vr, Wr, T8):
                # Ur,Vr,Wr are 2x the half-integer factors; divide by 2.
                Uf = [[x / 2 for x in row] for row in Ur.tolist()]
                Vf = [[x / 2 for x in row] for row in Vr.tolist()]
                Wf = [[x / 2 for x in row] for row in Wr.tolist()]
                return {"U": Uf, "V": Vf, "W": Wf}
    return best
