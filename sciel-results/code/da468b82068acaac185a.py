"""Rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
Uses block Strassen (rank 22) as baseline, then searches for lower rank via
vectorized BFGS optimization of the bilinear verification error.
"""
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 140.0  # seconds, leaves 10s margin before 150s kill


def _elapsed():
    return time.time() - _T0


def _remaining():
    return _DEADLINE - _elapsed()


def _build_tensors(n, m, p):
    """Precompute index arrays for vectorized error computation."""
    # Use np.indices to generate all combinations of indices safely
    # Order: i_U, j_U, j_V, k_V, i_W, k_W
    # Dimensions: n, m, m, p, n, p
    idx = np.indices((n, m, m, p, n, p))
    ia = idx[0].flatten()
    ja = idx[1].flatten()
    jb = idx[2].flatten()
    kb = idx[3].flatten()
    i = idx[4].flatten()
    k = idx[5].flatten()
    
    # Precompute the linear indices
    U_idx = ia * m + ja
    V_idx = jb * p + kb
    W_idx = k * n + i
    
    # Expected values: delta(ia,i) * delta(ja,jb) * delta(kb,k)
    expected = ((ia == i) & (ja == jb) & (kb == k)).astype(float)
    
    return U_idx, V_idx, W_idx, expected


def _bilinear_error_vec(U, V, W, n, m, p, U_idx, V_idx, W_idx, expected):
    """Vectorized bilinear error computation."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    
    # For each of the N bilinear terms: sum_r W[Widx,r] * U[Uidx,r] * V[Vidx,r]
    # We can compute this as: for each term, dot(W_row, U_row * V_row)
    # Vectorized: W_sel @ (U_sel * V_sel) per row... but that's still per-row.
    # Better: use einsum
    # W_sel: (N, r), U_sel: (N, r), V_sel: (N, r)
    # actual = sum_r W_sel[:,r] * U_sel[:,r] * V_sel[:,r] = np.einsum('nr,nr,nr->n', W_sel, U_sel, V_sel)
    W_sel = W[W_idx, :]
    U_sel = U[U_idx, :]
    V_sel = V[V_idx, :]
    actual = np.einsum('nr,nr,nr->n', W_sel, U_sel, V_sel)
    err = actual - expected
    return float(np.sum(err * err))


def _verify_full(U, V, W, n, m, p, tol=1e-8):
    """Full bilinear verification using vectorized computation."""
    U_idx, V_idx, W_idx, expected = _build_tensors(n, m, p)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    
    W_sel = W[W_idx, :]
    U_sel = U[U_idx, :]
    V_sel = V[V_idx, :]
    actual = np.einsum('nr,nr,nr->n', W_sel, U_sel, V_sel)
    return bool(np.all(np.abs(actual - expected) < tol))


def _rank22_list(n, m, p):
    """Construct rank-22 decomposition using block Strassen."""
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    # Block 1: 7 Strassen terms for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A: 0=A00, 1=A01, 3=A10, 4=A11
    # B: 0=B00, 1=B01, 4=B10, 5=B11
    # W: 0=C00, 2=C01, 1=C10, 3=C11

    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1

    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1

    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1

    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1

    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1

    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1

    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1

    # Block 2: 7 Strassen terms for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B: 2=B02, 3=B03, 6=B12, 7=B13
    # W: 4=C02, 6=C03, 5=C12, 7=C13

    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1

    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1

    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1

    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1

    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1

    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1

    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1

    # Block 3: 8 schoolbook terms for A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    # A: 2=A02, 5=A12
    # B: 8=B20, 9=B21, 10=B22, 11=B23
    for i, ai in [(0, 2), (1, 5)]:
        for k, bk in [(0, 8), (1, 9), (2, 10), (3, 11)]:
            U[ai][col] = 1
            V[bk][col] = 1
            W[k * n + i][col] = 1
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _to_dict(U, V, W):
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n

    # Precompute tensor indices for vectorized operations
    U_idx, V_idx, W_idx, expected = _build_tensors(n, m, p)

    # Start with rank-22
    best_U, best_V, best_W = _rank22_list(n, m, p)
    best_rank = 22

    # Verify
    if not _verify_full(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = n * m * p

    if _remaining() < 30.0:
        return _to_dict(best_U, best_V, best_W)

    from scipy.optimize import minimize

    def _try_rank(r, max_restarts=5, maxiter=300):
        """Try to find rank-r decomposition via BFGS optimization."""
        U22, V22, W22 = _rank22_list(n, m, p)
        U22 = np.array(U22, dtype=float)
        V22 = np.array(V22, dtype=float)
        W22 = np.array(W22, dtype=float)

        best_err = float('inf')
        best_params = None

        # Strategy 1: Drop columns from rank-22 (for r=21)
        if r == 21:
            for drop_col in range(22):
                if _remaining() < 20.0:
                    break
                keep = [c for c in range(22) if c != drop_col]
                U_init = U22[:, keep]
                V_init = V22[:, keep]
                W_init = W22[:, keep]
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

                def objective(x, U_idx=U_idx, V_idx=V_idx, W_idx=W_idx, expected=expected):
                    U = x[:nm * r].reshape(nm, r)
                    V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                    W = x[nm * r + mp * r:].reshape(pn, r)
                    return _bilinear_error_vec(U, V, W, n, m, p, U_idx, V_idx, W_idx, expected)

                try:
                    res = minimize(objective, x0, method='BFGS',
                                   options={'maxiter': maxiter, 'gtol': 1e-7})
                    if res.fun < best_err:
                        best_err = res.fun
                        best_params = res.x
                except Exception:
                    continue

        # Strategy 2: Random linear combinations of rank-22 columns
        for attempt in range(max_restarts):
            if _remaining() < 20.0:
                break
            
            # Initialize by mixing the 22 columns into r columns
            # This explores the subspace spanned by the rank-22 solution
            R = np.eye(22, r) + np.random.randn(22, r) * 0.1
            
            U_init = U22 @ R
            V_init = V22 @ R
            W_init = W22 @ R
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            def objective(x, U_idx=U_idx, V_idx=V_idx, W_idx=W_idx, expected=expected):
                U = x[:nm * r].reshape(nm, r)
                V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                W = x[nm * r + mp * r:].reshape(pn, r)
                return _bilinear_error_vec(U, V, W, n, m, p, U_idx, V_idx, W_idx, expected)

            try:
                res = minimize(objective, x0, method='BFGS',
                               options={'maxiter': maxiter, 'gtol': 1e-7})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception:
                continue

        if best_params is not None and best_err < 1e-6:
            # Polish pass for higher precision
            x = best_params
            def objective_polish(x, U_idx=U_idx, V_idx=V_idx, W_idx=W_idx, expected=expected):
                U = x[:nm * r].reshape(nm, r)
                V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                W = x[nm * r + mp * r:].reshape(pn, r)
                return _bilinear_error_vec(U, V, W, n, m, p, U_idx, V_idx, W_idx, expected)
            
            try:
                res_polish = minimize(objective_polish, x, method='BFGS',
                                      options={'maxiter': 200, 'gtol': 1e-12})
                if res_polish.fun < 1e-10:
                    x = res_polish.x
            except Exception:
                pass

            U = x[:nm * r].reshape(nm, r)
            V = x[nm * r:nm * r + mp * r].reshape(mp, r)
            W = x[nm * r + mp * r:].reshape(pn, r)
            if _verify_full(U, V, W, n, m, p, tol=1e-8):
                return U.tolist(), V.tolist(), W.tolist(), r
        return None

    # Try rank 20 first (known minimum for [2,3,4])
    if _remaining() > 60.0:
        result20 = _try_rank(20, max_restarts=8, maxiter=400)
        if result20 is not None:
            U20, V20, W20, r20 = result20
            return _to_dict(U20, V20, W20)

    # Try rank 21
    if _remaining() > 30.0:
        result21 = _try_rank(21, max_restarts=5, maxiter=300)
        if result21 is not None:
            U21, V21, W21, r21 = result21
            return _to_dict(U21, V21, W21)

    return _to_dict(best_U, best_V, best_W)
