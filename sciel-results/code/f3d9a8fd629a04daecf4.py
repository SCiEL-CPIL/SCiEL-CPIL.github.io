"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
START_TIME = time.time()
DEADLINE = 110.0  # safety margin before the 120s evaluation limit


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


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
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T


def _verify(U, V, W):
    # Exact check: sum_c U[x,c] V[y,c] W[z,c] == T[x,y,z] over the integers.
    T = _target_tensor()
    S = np.einsum('xc,yc,zc->xyz', U, V, W)
    if not np.allclose(S, np.round(S), atol=1e-6):
        return False
    return np.array_equal(np.round(S).astype(np.int64), T.astype(np.int64))


def _search(T, max_terms, rng):
    """Greedy rank-1 peeling: repeatedly subtract a rank-1 term that reduces
    the residual norm, using a truncated-CP (SVD) candidate plus random
    integer candidates. Returns a verified decomposition or None."""
    n, m, p = DIMENSIONS
    R = T.astype(np.float64).copy()
    R_norm_sq = float(np.sum(R * R))
    terms_U, terms_V, terms_W = [], [], []
    for step in range(max_terms):
        if _time_left() < 8.0:
            break
        best_term = None
        best_norm_sq = R_norm_sq
        # Truncated-CP candidate (SVD of the x-matricization).
        try:
            R2 = R.reshape(n * m, -1)
            _, _, Vt = np.linalg.svd(R2, full_matrices=False)
            u = Vt[0]
            w = np.tensordot(R, u.reshape(m * p, p * n), axes=([1, 2], [0, 1]))
            v = np.tensordot(R, w, axes=([2], [0]))
            for scale in (1, 2):
                ru = np.round(u * scale) / scale
                rv = np.round(v * scale) / scale
                rw = np.round(w * scale) / scale
                if not (np.any(ru) and np.any(rv) and np.any(rw)):
                    continue
                ip = float(np.einsum('x,y,z,xyz->', ru, rv, rw, R))
                nts = float(ru @ ru) * float(rv @ rv) * float(rw @ rw)
                nn = R_norm_sq - 2.0 * ip + nts
                if nn < best_norm_sq - 1e-6:
                    best_norm_sq = nn
                    best_term = (ru, rv, rw)
        except Exception:
            pass
        # Random integer rank-1 candidates (allow cancellation via -1,0,1).
        for _ in range(800):
            if _time_left() < 8.0:
                break
            u = rng.integers(-1, 2, size=n * m).astype(np.float64)
            v = rng.integers(-1, 2, size=m * p).astype(np.float64)
            w = rng.integers(-1, 2, size=p * n).astype(np.float64)
            ip = float(np.einsum('x,y,z,xyz->', u, v, w, R))
            nts = float(u @ u) * float(v @ v) * float(w @ w)
            nn = R_norm_sq - 2.0 * ip + nts
            if nn < best_norm_sq - 1e-6:
                best_norm_sq = nn
                best_term = (u, v, w)
        if best_term is None:
            break
        u, v, w = best_term
        R = R - np.einsum('x,y,z->xyz', u, v, w)
        R_norm_sq = float(np.sum(R * R))
        terms_U.append(u)
        terms_V.append(v)
        terms_W.append(w)
        if R_norm_sq < 1e-9:
            break
    if len(terms_U) == 0:
        return None
    U = np.column_stack(terms_U)
    V = np.column_stack(terms_V)
    W = np.column_stack(terms_W)
    if _verify(U, V, W):
        return U, V, W
    return None


def construct():
    U, V, W = _schoolbook()
    best = (U, V, W)
    best_rank = int(U.shape[1])
    T = _target_tensor()
    if _time_left() > 25.0:
        rng = np.random.default_rng(12345)
        found = _search(T, best_rank - 1, rng)
        if found is not None:
            fU, fV, fW = found
            if int(fU.shape[1]) < best_rank:
                best = found
                best_rank = int(fU.shape[1])
    bU, bV, bW = best

    def to_list(M):
        return [[float(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]

    return {"U": to_list(bU), "V": to_list(bV), "W": to_list(bW)}
