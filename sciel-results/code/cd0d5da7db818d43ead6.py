"""Combined program: structural seed + refinement search for lower-rank decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before the 150 s wall-clock kill
_TOL = 1e-7


def _matmul_tensor(n, m, p):
    """Return the matrix-multiplication tensor T with T[i*m+j, j*p+k, k*n+i] = 1."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, tol=_TOL):
    """Check sum_r U[:,r]*V[:,r]*W[:,r] == T via matrix slices (fast)."""
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    for j in range(m):
        u = U[:j + 1: m]          # rows i*m+j for all i  (n,)
        v = V[j * p:(j + 1) * p]  # rows j*p+k for all k  (p,)
        mat = u[:, None] * v[None, :]  # (n, p)
        w = W.reshape(p, n)       # W[k*n+i]
        if np.max(np.abs(mat - mat @ w)) > tol:
            return False
    return True


def _refine(U, V, W, T, n, m, p, max_iter=60, tol=_TOL):
    """Least-squares refinement of a rank-R decomposition. Returns (U,V,W,ok)."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    R = U.shape[1]
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    wmat = W.reshape(p, n)

    def rhs_for(j):
        mat = np.zeros((n, p), dtype=np.float64)
        for r in range(R):
            mat += U[:j + 1: m, r] * V[j * p:(j + 1) * p, r]
        return mat

    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        for j in range(m):
            A = rhs_for(j)
            for i in range(n):
                b = T[i * m + j, j * p:(j + 1) * p, i * n + i].copy()
                sol, *_ = np.linalg.lstsq(A[i], b, rcond=None)
                wmat[:, i] = sol
            W = wmat.reshape(p * n)
        for j in range(m):
            A = wmat.T @ U[:j + 1: m]  # (p, R)
            for k in range(p):
                b = T[j * p + k, j * p:(j + 1) * p, k * n:(k + 1) * n].copy()
                sol, *_ = np.linalg.lstsq(A[k], b, rcond=None)
                V[j * p + k] = sol
        for j in range(m):
            A = wmat @ V[j * p:(j + 1) * p]  # (n, R)
            for i in range(n):
                b = T[i * m + j, j * p:(j + 1) * p, i * n + i].copy()
                sol, *_ = np.linalg.lstsq(A[i], b, rcond=None)
                U[:j + 1: m, :] = sol
        if _verify(U, V, W, T, tol):
            return U, V, W, True
    return U, V, W, _verify(U, V, W, T, tol)


def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    # Guaranteed fallback: schoolbook decomposition of rank n*m*p.
    rank_fb = n * m * p
    U_fb = np.zeros((dim_u, rank_fb), dtype=np.int64)
    V_fb = np.zeros((dim_v, rank_fb), dtype=np.int64)
    W_fb = np.zeros((dim_w, rank_fb), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    best = {"U": U_fb, "V": V_fb, "W": W_fb, "rank": rank_fb}

    T = _matmul_tensor(n, m, p)

    # Search for a strictly lower rank via random init + least-squares refinement.
    rng = np.random.default_rng(12345)
    for R in range(30, rank_fb):  # 30 .. 47
        if time.time() > _DEADLINE:
            break
        attempts = 4
        found = False
        for _ in range(attempts):
            if time.time() > _DEADLINE:
                break
            U = rng.standard_normal((dim_u, R))
            V = rng.standard_normal((dim_v, R))
            W = rng.standard_normal((dim_w, R))
            U, V, W, ok = _refine(U, V, W, T, n, m, p)
            if ok:
                best = {"U": U, "V": V, "W": W, "rank": R}
                found = True
                break
        if found:
            break

    return {"U": best["U"].tolist(), "V": best["V"].tolist(), "W": best["W"].tolist()}
# EVOLVE-BLOCK-END
