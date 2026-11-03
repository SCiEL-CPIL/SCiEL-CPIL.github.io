"""Rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.

Structural baseline: Kronecker-embedded Strassen (rank 65) + naive 5th column.
Refinement: iterative null-space elimination — detect columns whose tensor
contribution lies in the linear span of the other columns' contributions and
prune them, then attempt to re-express the residual to close the gap.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
DEADLINE_MARGIN = 15.0  # seconds to keep under the 150s kill


# ------------------------- verification -------------------------
def _verify(U, V, W, tol=1e-9):
    """Check T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    n, m, p = N, M, P
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    rank = Ur.shape[2]
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        for j in range(m):
            T[:, j, :] += np.outer(uc[:, j], vc[j, :]) * wc.T
    err = np.max(np.abs(T - 1.0))
    return err < tol, err


def _round_halfint(A):
    return np.round(2.0 * A) / 2.0


# ------------------- Strassen 2x2x2 primitive -------------------
def _strassen_2x2x2_terms():
    """7 Strassen terms as sparse (idx, val) lists for U, V, W.
    U idx: (i,j)->i*2+j ; V idx: (j,k)->j*2+k ; W idx: (k,i)->k*2+i
    """
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),       # (a+d)(e+h)
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),            # (c+d)e
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),            # a(f-h)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),            # d(g-e)
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),            # (a+b)h
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),            # (c-a)(e+f)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),            # (b-d)(g+h)
    ]


# ------------------ Kronecker-Strassen baseline ------------------
def _kron_strassen_baseline():
    """4x4x4 via Kronecker of two 2x2x2 Strassen (rank 49) + naive 5th col (16)."""
    n, m, p = N, M, P
    U_cols, V_cols, W_cols = [], [], []

    def add_term(u, v, w):
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

    terms = _strassen_2x2x2_terms()
    for u1s, v1s, w1s in terms:
        for u2s, v2s, w2s in terms:
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            for (i1, j1), val1 in [(ix // 2, ix % 2) for ix, _ in u1s] and [(ix // 2, ix % 2, val) for ix, val in u1s]:
                pass
            # U: (i1*2+i2, j1*2+j2)
            for (ix1, val1) in u1s:
                i1, j1 = ix1 // 2, ix1 % 2
                for (ix2, val2) in u2s:
                    i2, j2 = ix2 // 2, ix2 % 2
                    u[(i1*2+i2) * m + (j1*2+j2)] = val1 * val2
            # V: (j1*2+j2, k1*2+k2)
            for (ix1, val1) in v1s:
                j1, k1 = ix1 // 2, ix1 % 2
                for (ix2, val2) in v2s:
                    j2, k2 = ix2 // 2, ix2 % 2
                    v[(j1*2+j2) * p + (k1*2+k2)] = val1 * val2
            # W: (k1*2+k2, i1*2+i2)
            for (ix1, val1) in w1s:
                k1, i1 = ix1 // 2, ix1 % 2
                for (ix2, val2) in w2s:
                    k2, i2 = ix2 // 2, ix2 % 2
                    w[(k1*2+k2) * n + (i1*2+i2)] = val1 * val2
            add_term(u, v, w)

    # naive 5th column (k=4)
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[i*m+j] = 1
            v = np.zeros(m*p); v[j*p+4] = 1
            w = np.zeros(p*n); w[4*n+i] = 1
            add_term(u, v, w)

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    return U, V, W


# ------------- tensor contribution of one column ----------------
def _col_tensor(U, V, W, c):
    n, m, p = N, M, P
    return (U[:, c].reshape(n, m)
            * V[:, c].reshape(m, p)
            * W[:, c].reshape(p, n)).T.reshape(-1)  # indexed (k, j, i) -> flat


def _all_col_tensors(U, V, W):
    r = U.shape[1]
    return np.array([_col_tensor(U, V, W, c) for c in range(r)])


# ----------------- null-space elimination core -------------------
def _eliminate_dependent_columns(U, V, W, deadline):
    """Iteratively remove columns whose tensor contribution is in the span of
    the others. Returns pruned (U, V, W)."""
    r = U.shape[1]
    tensors = _all_col_tensors(U, V, W)  # (r, n*m*p)
    keep = list(range(r))
    changed = True
    while changed and (time.time() < deadline):
        changed = False
        # Build matrix of kept tensors; find columns that are dependent
        T_keep = tensors[keep]
        # compute rank via SVD
        try:
            _, s, Vt = np.linalg.svd(T_keep, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        tol = 1e-8 * (s[0] if len(s) else 1.0)
        rank = int(np.sum(s > tol))
        if rank < len(keep):
            # find a dependent column: the one with smallest residual when
            # projecting onto the span of the others is removed.
            # Use: for each column, residual norm after projecting onto span
            # of the others. Remove the one with smallest residual.
            best_c = None
            best_res = np.inf
            Uq = U[:, keep]
            Vq = V[:, keep]
            Wq = W[:, keep]
            for li in range(len(keep)):
                others = [x for x in range(len(keep)) if x != li]
                if len(others) == 0:
                    continue
                T_others = T_keep[others]
                # project T_keep[li] onto span of T_others
                try:
                    proj, *_ = np.linalg.lstsq(T_others.T, T_keep[li], rcond=None)
                    resid = T_keep[li] - T_others @ proj
                    res = np.linalg.norm(resid)
                except np.linalg.LinAlgError:
                    continue
                if res < best_res:
                    best_res = res
                    best_c = li
            if best_c is not None and best_res < 1e-7:
                # drop keep[best_c]
                dropped = keep[best_c]
                keep = [x for x in keep if x != dropped]
                changed = True
        else:
            break
    if len(keep) < r:
        U = U[:, keep]; V = V[:, keep]; W = W[:, keep]
    return U, V, W


# ----------------- re-express residual to close gap ----------------
def _try_close_residue(U, V, W, deadline):
    """After pruning, some terms may be missing. Try least-squares re-fit of
    pruned factor entries (with half-int rounding) to restore exactness, then
    re-eliminate. Returns (U,V,W,ok)."""
    n, m, p = N, M, P
    r = U.shape[1]
    if r == 0:
        return U, V, W, False
    # ALS-style re-fit for a few iterations
    Uc, Vc, Wc = U.copy(), V.copy(), W.copy()
    for it in range(30):
        if time.time() > deadline:
            break
        # update U
        Un = Uc.copy()
        for i in range(n):
            for j in range(m):
                Mm = np.zeros((p, r))
                for k in range(p):
                    Mm[k, :] = Vc[j*p+k, :] * Wc[k*n+i, :]
                try:
                    sol, *_ = np.linalg.lstsq(Mm.T, np.ones(p), rcond=None)
                    Un[i*m+j, :] = sol
                except np.linalg.LinAlgError:
                    pass
        Uc = Un
        # update V
        Vn = Vc.copy()
        for j in range(m):
            for k in range(p):
                Mm = np.zeros((n, r))
                for i in range(n):
                    Mm[i, :] = Uc[i*m+j, :] * Wc[k*n+i, :]
                try:
                    sol, *_ = np.linalg.lstsq(Mm.T, np.ones(n), rcond=None)
                    Vn[j*p+k, :] = sol
                except np.linalg.LinAlgError:
                    pass
        Vc = Vn
        # update W
        Wn = Wc.copy()
        for k in range(p):
            for i in range(n):
                Mm = np.zeros((m, r))
                for j in range(m):
                    Mm[j, :] = Uc[i*m+j, :] * Vc[j*p+k, :]
                try:
                    sol, *_ = np.linalg.lstsq(Mm.T, np.ones(m), rcond=None)
                    Wn[k*n+i, :] = sol
                except np.linalg.LinAlgError:
                    pass
        Wc = Wn
        # round and check
        Ur = _round_halfint(Uc); Vr = _round_halfint(Vc); Wr = _round_halfint(Wc)
        ok, err = _verify(Ur, Vr, Wr)
        if ok:
            return Ur, Vr, Wr, True
    return Uc, Vc, Wc, False


# --------------------------- construct ---------------------------
def construct():
    start = time.time()
    deadline = start + 150.0 - DEADLINE_MARGIN

    # 1. structural baseline
    U, V, W = _kron_strassen_baseline()
    ok, err = _verify(U, V, W)
    if not ok:
        # fallback to schoolbook
        n, m, p = N, M, P
        rank = n*m*p
        Ub = np.zeros((n*m, rank)); Vb = np.zeros((m*p, rank)); Wb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Ub[i*m+j, col] = 1.0
                    Vb[j*p+k, col] = 1.0
                    Wb[k*n+i, col] = 1.0
                    col += 1
        return {"U": Ub.tolist(), "V": Vb.tolist(), "W": Wb.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    # 2. null-space elimination
    U, V, W = _eliminate_dependent_columns(U, V, W, deadline)
    ok, err = _verify(U, V, W)
    if ok and U.shape[1] < best_rank:
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]

    # 3. re-express residue to close any gap, then re-eliminate
    while time.time() < deadline - 3:
        U2, V2, W2, ok = _try_close_residue(U, V, W, deadline)
        if ok and U2.shape[1] < best_rank:
            best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()
            best_rank = U2.shape[1]
        # re-eliminate on the (possibly re-fitted) factors
        U3, V3, W3 = _eliminate_dependent_columns(U2, V2, W2, deadline)
        ok3, _ = _verify(U3, V3, W3)
        if ok3 and U3.shape[1] < best_rank:
            best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
            best_rank = U3.shape[1]
            U, V, W = U3, V3, W3
        else:
            # no further progress; stop
            break

    # final verification; fall back to best valid if needed
    ok, _ = _verify(best_U, best_V, best_W)
    if not ok:
        U, V, W = _kron_strassen_baseline()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
