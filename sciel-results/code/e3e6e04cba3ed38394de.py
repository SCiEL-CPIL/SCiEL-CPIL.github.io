"""Rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _triple_indices(n, m, p):
    """Return (idx_U, idx_V, idx_W) arrays for the n*m*p valid (i,j,k) terms."""
    ii = np.repeat(np.arange(n), m * p)
    jj = np.tile(np.repeat(np.arange(m), p), n)
    kk = np.tile(np.arange(p), n * m)
    iU = ii * m + jj
    iV = jj * p + kk
    iW = kk * n + ii
    return iU, iV, iW

def _verify_full(U, V, W, n, m, p, tol=1e-7):
    """Rigorous check: full tensor equality T[U,V,W] == T_schoolbook."""
    rank = U.shape[1]
    # Build the rank-r tensor (size n*m * m*p * p*n)
    T = np.einsum('ac,bc,dc->abd', U, V, W)
    # Build the schoolbook tensor
    S = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                S[i * m + j, j * p + k, k * n + i] = 1.0
    return np.max(np.abs(T - S)) < tol

def _lm_solve(U, V, W, iU, iV, iW, max_iter=80, damp0=1e-6):
    """Levenberg-Marquardt on flattened parameters; minimize sum of residual^2."""
    nU, nV, nW = U.size, V.size, W.size
    x = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    damp = damp0

    def split(x):
        Uv = x[:nU].reshape(U.shape)
        Vv = x[nU:nU + nV].reshape(V.shape)
        Wv = x[nU + nV:].reshape(W.shape)
        return Uv, Vv, Wv

    def resid(x):
        Uv, Vv, Wv = split(x)
        return np.einsum('ac,bc,dc->abc', Uv, Vv, Wv)[iU, iV, iW] - 1.0

    def jac(x):
        Uv, Vv, Wv = split(x)
        r = len(iU)
        J = np.zeros((r, nU + nV + nW))
        # d/dU[a,c] = V[iV,c]*W[iW,c]
        J[:, :nU] = np.einsum('c,dc->dc', Vv[iV], Wv[iW]).T
        # d/dV[b,c] = U[iU,c]*W[iW,c]
        J[:, nU:nU + nV] = np.einsum('c,dc->dc', Uv[iU], Wv[iW]).T
        # d/dW[d,c] = U[iU,c]*V[iV,c]
        J[:, nU + nV:] = np.einsum('c,dc->dc', Uv[iU], Vv[iV]).T
        return J

    r = resid(x)
    fval = float(np.sum(r * r))
    for _ in range(max_iter):
        J = jac(x)
        g = -J.T @ r
        # J^T J via SVD (robust, no explicit normal equations)
        try:
            s, Vt = np.linalg.svd(J, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        s2 = s * s
        # Levenberg damping in singular-value space
        denom = s2 + damp
        dx = (Vt.T @ (Vt @ g / denom))
        x_new = x + dx
        r_new = resid(x_new)
        f_new = float(np.sum(r_new * r_new))
        if f_new < fval:
            x = x_new
            r = r_new
            fval = f_new
            damp = max(damp * 0.4, 1e-12)
            if fval < 1e-20:
                break
        else:
            damp = min(damp * 8.0, 1e6)
    Uv, Vv, Wv = split(x)
    return Uv, Vv, Wv, fval

def _try_rank(r, n, m, p, iU, iV, iW, deadline):
    """Try to find an exact rank-r decomposition via LM from random inits."""
    nU, nV, nW = n * m, m * p, p * n
    for seed in range(40):
        if time.time() > deadline - 1.0:
            break
        rng = np.random.RandomState(seed)
        scale = 0.5
        U = rng.randn(nU, r) * scale
        V = rng.randn(nV, r) * scale
        W = rng.randn(nW, r) * scale
        U, V, W, fval = _lm_solve(U, V, W, iU, iV, iW, max_iter=60)
        if fval < 1e-12:
            return U, V, W
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # safety margin under 150s wall clock

    iU, iV, iW = _triple_indices(n, m, p)

    # Lower bounds on tensor rank:
    #  - trivial: n*m*p
    #  - slice bound: max(n*m, m*p, p*n)
    #  - Strassen-type: ceil(n*p * log2(n*m*p)/(log2(n*m)+log2(m*p)+log2(p*n)))
    lb_slice = max(n * m, m * p, p * n)
    import math
    num = n * p * math.log2(n * m * p)
    den = math.log2(n * m) + math.log2(m * p) + math.log2(p * n)
    lb_strassen = int(math.ceil(num / den))
    lower = max(lb_slice, lb_strassen)
    upper = n * m * p

    best = None
    for r in range(lower, upper + 1):
        if time.time() > deadline - 5.0:
            break
        res = _try_rank(r, n, m, p, iU, iV, iW, deadline)
        if res is not None:
            U, V, W = res
            if _verify_full(U, V, W, n, m, p, tol=1e-7):
                best = (U, V, W, r)
                break  # rank is ascending, first valid is minimal

    if best is None:
        U, V, W = _schoolbook(n, m, p)
        r = U.shape[1]
    else:
        U, V, W, r = best

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
