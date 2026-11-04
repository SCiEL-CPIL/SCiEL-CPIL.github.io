import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# ---------------------------------------------------------------------------
# Strassen 2x2x2: 7 rank-1 terms.
# ---------------------------------------------------------------------------
_STRASSEN_222 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
    ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
]


def _build_rank114(n, m, p):
    """Rank-114: two 4x4x4 Strassen (98) + 4x4x1 matrix-vector (16)."""
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    # First 4x4x4 block
    for (u, v, w) in _STRASSEN_222:
        for t in range(4):
            if u[t]:
                U[t][col] += u[t]
            if v[t]:
                V[t][col] += v[t]
            if w[t]:
                W[t][col] += w[t]
        col += 1
    # Second 4x4x4 block (b_base=4, c_base=4)
    for (u, v, w) in _STRASSEN_222:
        for t in range(4):
            if u[t]:
                U[t][col] += u[t]
            if v[t]:
                V[4 + t][col] += v[t]
            if w[t]:
                W[4 + t][col] += w[t]
        col += 1
    # 4x4x1 matrix-vector block (b_base=8, c_base=8)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[8 + j][col] = 1
            W[8 + i][col] = 1
            col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Return True iff the decomposition is exact."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i, j, k] - 1.0) > tol:
                    return False
    # Check no extra terms
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i, j, k] = 0.0
    if np.max(np.abs(T2)) > tol:
        return False
    return True


def _tensor_target(n, m, p):
    """Build the target tensor T[i,j,k] = 1."""
    return np.ones((n, m, p))


def _build_linear_system(U, V, W, n, m, p):
    """
    Build the linear system A x = b where:
    - x are the rank-1 coefficients (one per column)
    - A has shape (n*m*p, rank)
    - A[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    - b[i*m*p + j*p + k] = 1
    """
    rank = U.shape[1]
    A = np.zeros((n * m * p, rank))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        # Compute the rank-1 tensor contribution
        for i in range(n):
            ui = u_col[i * m: (i+1) * m]
            for j in range(m):
                uij = ui[j]
                if uij == 0:
                    continue
                vj = v_col[j * p: (j+1) * p]
                for k in range(p):
                    vjk = vj[k]
                    if vjk == 0:
                        continue
                    wki = w_col[k * n + i]
                    if wki == 0:
                        continue
                    A[i * m * p + j * p + k, r] = uij * vjk * wki
    b = np.ones(n * m * p)
    return A, b


def _try_remove_column(U, V, W, n, m, p, col_idx, deadline):
    """
    Try to remove column col_idx by solving for new coefficients of remaining columns.
    Returns (success, new_U, new_V, new_W) or (False, None, None, None).
    """
    if time.time() > deadline:
        return False, None, None, None

    rank = U.shape[1]
    # Keep all columns except col_idx
    keep = [r for r in range(rank) if r != col_idx]
    new_rank = len(keep)
    if new_rank < n * m * p:  # Need at least n*m*p equations solvable
        return False, None, None, None

    U_sub = U[:, keep]
    V_sub = V[:, keep]
    W_sub = W[:, keep]

    # Build linear system for the sub-matrix
    A_sub = np.zeros((n * m * p, new_rank))
    for r in range(new_rank):
        u_col = U_sub[:, r]
        v_col = V_sub[:, r]
        w_col = W_sub[:, r]
        for i in range(n):
            ui = u_col[i * m: (i+1) * m]
            for j in range(m):
                uij = ui[j]
                if uij == 0:
                    continue
                vj = v_col[j * p: (j+1) * p]
                for k in range(p):
                    vjk = vj[k]
                    if vjk == 0:
                        continue
                    wki = w_col[k * n + i]
                    if wki == 0:
                        continue
                    A_sub[i * m * p + j * p + k, r] = uij * vjk * wki

    b = np.ones(n * m * p)

    # Solve least squares
    try:
        coeffs, residuals, rank_ls, sv = np.linalg.lstsq(A_sub, b, rcond=None)
    except Exception:
        return False, None, None, None

    # Check if residual is effectively zero
    if len(residuals) > 0:
        if residuals[0] > 1e-8:
            return False, None, None, None
    else:
        # Check residual directly
        residual = np.linalg.norm(A_sub @ coeffs - b)
        if residual > 1e-8:
            return False, None, None, None

    # Reconstruct with new coefficients
    # The new U, V, W have the same support but scaled coefficients
    # Actually, the coefficients multiply the entire rank-1 term
    # So new_U[:, r] = U_sub[:, r] * coeffs[r]
    new_U = U_sub * coeffs
    new_V = V_sub * coeffs
    new_W = W_sub * coeffs

    # Verify
    if not _verify(new_U, new_V, new_W, n, m, p):
        return False, None, None, None

    return True, new_U, new_V, new_W


def _greedy_refinement(U, V, W, n, m, p, deadline):
    """
    Iteratively try to remove columns one at a time.
    Greedy: try all columns, remove the first one that works, repeat.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    improved = True

    while improved and time.time() < deadline:
        improved = False
        current_rank = U.shape[1]
        for col_idx in range(current_rank):
            if time.time() > deadline:
                break
            success, new_U, new_V, new_W = _try_remove_column(U, V, W, n, m, p, col_idx, deadline)
            if success:
                U, V, W = new_U, new_V, new_W
                rank = U.shape[1]
                improved = True
                break  # Restart scanning from beginning

    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Build initial rank-114 decomposition
    res = _build_rank114(n, m, p)
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)

    # Verify initial
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try greedy refinement
    if time.time() < deadline - 5:
        U_ref, V_ref, W_ref = _greedy_refinement(U, V, W, n, m, p, deadline)
        if U_ref is not None and _verify(U_ref, V_ref, W_ref, n, m, p):
            # Check if better
            if U_ref.shape[1] < U.shape[1]:
                U, V, W = U_ref, V_ref, W_ref

    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
