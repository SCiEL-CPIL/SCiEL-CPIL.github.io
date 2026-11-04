"""Rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
_START = time.time()
_DEADLINE = 150.0


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :] * V[j * p + k, :], W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _to_frac(M, tol=1e-9):
    """Convert a float matrix to exact fractions (half-integers)."""
    M = np.array(M, dtype=float)
    M2 = np.round(M * 2) / 2
    if np.allclose(M, M2, atol=tol):
        M = M2
    out = []
    for row in M:
        r = []
        for x in row:
            f = Fraction(str(round(float(x), 12)))
            r.append(f)
        out.append(r)
    return out


def _err_vec(U, V, W, n, m, p):
    """Return the n*m*p error vector (val-1) for the decomposition."""
    r = U.shape[1]
    e = np.empty(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                e[idx] = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i]) - 1.0
                idx += 1
    return e


def _maxerr(e):
    return float(np.max(np.abs(e))) if e.size else float('inf')


def _als_decompose(n, m, p, rank, U, V, W, n_iter=60, seed=42,
                   tol=1e-11, ridge=1e-10):
    """Convexity-preserving alternating least squares.

    Each update solves a linear least-squares problem (convex in the solved
    block), which is far more robust than the non-convex update in the seed.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    e = _err_vec(U, V, W, n, m, p)
    prev = _maxerr(e)

    for it in range(n_iter):
        if time.time() > _DEADLINE - 15:
            break

        # --- Update U: for each (i,j), A[k,c]=V[jp+k,c]*W[kn+i,c], b[k]=1 ---
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k] * W[k * n + i]
                x, *_ = np.linalg.lstsq(A + ridge * np.eye(rank), np.ones(p), rcond=None)
                U[i * m + j] = x

        # --- Update V: for each (j,k), A[i,c]=U[im+j,c]*W[kn+i,c], b[i]=1 ---
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j] * W[k * n + i]
                x, *_ = np.linalg.lstsq(A + ridge * np.eye(rank), np.ones(n), rcond=None)
                V[j * p + k] = x

        # --- Update W: for each (k,i), A[j,c]=U[im+j,c]*V[jp+k,c], b[j]=1 ---
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j] * V[j * p + k]
                x, *_ = np.linalg.lstsq(A + ridge * np.eye(rank), np.ones(m), rcond=None)
                W[k * n + i] = x

        e = _err_vec(U, V, W, n, m, p)
        cur = _maxerr(e)
        if cur < tol:
            break
        if cur > 1.5 * prev + 1e-6:
            # diverging: stop early, keep last good
            e = _err_vec(U, V, W, n, m, p)
            break
        prev = cur

    e = _err_vec(U, V, W, n, m, p)
    return U, V, W, e


def _deflate(U, V, W, n, m, p, e, tol=1e-10):
    """Exact rank reduction: if one tensor factor has a zero column, drop it."""
    for _ in range(80):
        if time.time() > _DEADLINE - 15:
            break
        r = U.shape[1]
        if r == 0:
            break
        # find a factor with a (numerically) zero column
        cols = [np.abs(U).sum(axis=0), np.abs(V).sum(axis=0), np.abs(W).sum(axis=0)]
        col = None
        for c in range(r):
            if cols[0][c] < 1e-13 or cols[1][c] < 1e-13 or cols[2][c] < 1e-13:
                col = c
                break
        if col is None:
            break
        keep = [c for c in range(r) if c != col]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        e = _err_vec(U, V, W, n, m, p)
        if _maxerr(e) > tol:
            # should not happen; restore
            break
    return U, V, W, e


def _strassen_init(n, m, p, rank):
    """Structured prior: schoolbook (rank 45) for the 3x3x3 sub-block plus
    random columns for the remaining k, then truncated to `rank`."""
    rng = np.random.RandomState(7)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    # schoolbook for k in 0..2 (9 terms)
    for c, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(3)]):
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    # fill remaining columns with small random noise (gives ALS degrees of freedom)
    if rank > 9:
        U[:, 9:] = rng.randn(n * m, rank - 9) * 0.05
        V[:, 9:] = rng.randn(m * p, rank - 9) * 0.05
        W[:, 9:] = rng.randn(p * n, rank - 9) * 0.05
    return U, V, W


def _random_init(n, m, p, rank, seed):
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    return U, V, W


def _try_rank(n, m, p, rank, n_trials=4, n_iter=60):
    best = None
    best_err = float('inf')
    for trial in range(n_trials):
        if time.time() > _DEADLINE - 15:
            break
        if trial == 0:
            U, V, W = _strassen_init(n, m, p, rank)
        else:
            U, V, W = _random_init(n, m, p, rank, seed=trial * 1000 + 42)
        U, V, W, e = _als_decompose(n, m, p, rank, U, V, W, n_iter=n_iter, seed=trial)
        U, V, W, e = _deflate(U, V, W, n, m, p, e)
        err = _maxerr(e)
        if err < best_err:
            best_err = err
            best = (U, V, W)
        if err < 1e-9:
            return U, V, W, err
    if best is None:
        return None, None, None, float('inf')
    return best[0], best[1], best[2], best_err


def _schoolbook(n, m, p):
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


def construct():
    n, m, p = DIMENSIONS
    # Try ranks below schoolbook (45) from low to high.
    for rank in [20, 24, 27, 30, 33, 36, 39, 42]:
        if time.time() > _DEADLINE - 15:
            break
        U, V, W, err = _try_rank(n, m, p, rank, n_trials=4, n_iter=60)
        if err < 1e-8 and _verify(U, V, W, n, m, p):
            Uf = _to_frac(U)
            Vf = _to_frac(V)
            Wf = _to_frac(W)
            return {"U": Uf, "V": Vf, "W": Wf}
    # Fall back to schoolbook (rank 45)
    return _schoolbook(n, m, p)
