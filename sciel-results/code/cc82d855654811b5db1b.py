import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
_DEADLINE_START = None


def _ensure_start():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    return _DEADLINE_START


def _time_left(safety=8.0):
    s = _ensure_start()
    return 150.0 - (time.time() - s) - safety


def _as_list(x):
    return [[float(v) for v in row] for row in np.asarray(x, dtype=float)]


def _check(U, V, W, n, m, p, tol=1e-6):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    T = (U.reshape(n, m, r) * V.T[m, :, r]).sum(axis=2) * W.T[p, :, r]
    T = T.transpose(2, 0, 1)  # (i, j, k)
    T = T.reshape(n * m, p)
    ref = np.zeros((n * m, p))
    for i in range(n):
        for j in range(m):
            ref[i * m + j, j] = 1.0
    if not np.allclose(T, ref, atol=tol):
        return False, float(np.max(np.abs(T - ref)))
    # half-integer test
    if not np.allclose(T * 2.0, np.round(T * 2.0), atol=1e-9):
        return False, float(np.max(np.abs(T * 2.0 - np.round(T * 2.0))))
    return True, 0.0


def _schoolbook(n, m, p):
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


def _strassen_block(n, m, p, n0, m0, p0, i0, j0, k0, tol=1e-6):
    """Rank-7 Strassen block on rows [i0,i0+n0) x cols [k0,k0+p0)."""
    if n0 != 2 or m0 != 2 or p0 != 4:
        return None
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))
    idx = lambda i, j, k: (i0 + i, j0 + j, k0 + k)
    m1 = idx(0, 0, 0); m2 = idx(0, 0, 1); m3 = idx(0, 1, 0); m4 = idx(0, 1, 1)
    m5 = idx(1, 0, 0); m6 = idx(1, 0, 1); m7 = idx(1, 1, 0); m8 = idx(1, 1, 1)

    # M1=(A11+A22)(B11+B22) -> B12, B21
    U[idx(0,0),0]=1; U[idx(1,1),0]=1
    V[idx(0,0),0]=1; V[idx(1,1),0]=1
    W[idx(0,1),0]=1; W[idx(1,0),0]=1
    # M2=(A21-A11)B11 -> B21
    U[idx(1,0),1]=-1; U[idx(0,0),1]=1
    V[idx(0,0),1]=1
    W[idx(1,0),1]=1
    # M3=A11(B21-B11) -> B12
    U[idx(0,0),2]=1
    V[idx(0,1),2]=-1; V[idx(1,0),2]=1
    W[idx(0,1),2]=1
    # M4=A22(B12+B22) -> B12
    U[idx(1,1),3]=1
    V[idx(0,0),3]=1; V[idx(1,1),3]=1
    W[idx(0,1),3]=1
    # M5=(A11+A12)B22 -> B11
    U[idx(0,0),4]=1; U[idx(0,1),4]=1
    V[idx(1,1),4]=1
    W[idx(0,0),4]=1
    # M6=(A21-A11)(B11+B12) -> B22
    U[idx(1,0),5]=-1; U[idx(0,0),5]=1
    V[idx(0,0),5]=1; V[idx(0,1),5]=1
    W[idx(1,1),5]=1
    # M7=(A12-A22)(B21+B22) -> B11
    U[idx(0,1),6]=1; U[idx(1,1),6]=-1
    V[idx(1,0),6]=1; V[idx(1,1),6]=1
    W[idx(0,0),6]=1

    ok, _ = _check(U, V, W, n, m, p, tol)
    if not ok:
        return None
    return U, V, W


def _place_block(best, block, n, m, p, n0, m0, p0, i0, j0, k0, tol=1e-6):
    """Place a rank-7 block into the best decomposition, overwriting its sub-region."""
    U, V, W = block
    best_U = best[0].copy()
    best_V = best[1].copy()
    best_W = best[2].copy()
    for c in range(7):
        for i in range(n0):
            best_U[(i0 + i) * m + j0, c] = U[(i) * m + j0, c]
            best_U[(i0 + i) * m + (j0 + 1), c] = U[(i) * m + (j0 + 1), c]
        for j in range(m0):
            best_V[(j0 + j) * p + k0, c] = V[(j) * p + k0, c]
            best_V[(j0 + j) * p + (k0 + 1), c] = V[(j) * p + (k0 + 1), c]
        for k in range(p0):
            best_W[(k0 + k) * n + i0, c] = W[(k) * n + i0, c]
            best_W[(k0 + k) * n + (i0 + 1), c] = W[(k) * n + (i0 + 1), c]
    ok, _ = _check(best_U, best_V, best_W, n, m, p, tol)
    if not ok:
        return None
    return (best_U, best_V, best_W)


def construct():
    _ensure_start()
    n, m, p = DIMENSIONS

    # Start from schoolbook (guaranteed valid, rank 63)
    best_U, best_V, best_W = _schoolbook(n, m, p)
    ok, _ = _check(best_U, best_V, best_W, n, m, p)
    best_rank = n * m * p
    if not ok:
        return {"U": _as_list(best_U), "V": _as_list(best_V), "W": _as_list(best_W)}

    # Try to embed a rank-7 Strassen block (reduces rank from 63 to 63 - 16 + 7 = 54)
    block = _strassen_block(n, m, p, 2, 2, 4, 0, 0, 0)
    if block is not None and _time_left() > 5.0:
        placed = _place_block((best_U, best_V, best_W), block, n, m, p, 2, 2, 4, 0, 0, 0)
        if placed is not None:
            best_U, best_V, best_W = placed
            best_rank = 54

    return {"U": _as_list(best_U), "V": _as_list(best_V), "W": _as_list(best_W)}
