"""Matrix multiplication 3x4 by 4x6: Strassen base + linear-algebra rank reduction."""
import time
import itertools

import numpy as np

n, m, p = 3, 4, 6

_START_TIME = time.time()
_DEADLINE = 143.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _time_ok():
    return _time_left() > 10.0


def _build_base():
    """Build the rank-66 base decomposition. Returns (U, V, W) as numpy int arrays."""
    rank = 66
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        a = i_base * m + j_base
        b = i_base * m + (j_base + 1)
        c = (i_base + 1) * m + j_base
        d = (i_base + 1) * m + (j_base + 1)
        e = j_base * p + k_base
        f = j_base * p + (k_base + 1)
        g = (j_base + 1) * p + k_base
        h = (j_base + 1) * p + (k_base + 1)
        P = k_base * n + i_base
        Q = (k_base + 1) * n + i_base
        R = k_base * n + (i_base + 1)
        S = (k_base + 1) * n + (i_base + 1)

        # P1 = (a+d)(e+h)
        U[a, col] = 1; U[d, col] = 1
        V[e, col] = 1; V[h, col] = 1
        W[P, col] = 1; W[S, col] = 1
        col += 1
        # P2 = (c+d)e
        U[c, col] = 1; U[d, col] = 1
        V[e, col] = 1
        W[R, col] = 1; W[S, col] = -1
        col += 1
        # P3 = a(f-h)
        U[a, col] = 1
        V[f, col] = 1; V[h, col] = -1
        W[Q, col] = 1; W[S, col] = 1
        col += 1
        # P4 = d(g-e)
        U[d, col] = 1
        V[g, col] = 1; V[e, col] = -1
        W[P, col] = 1; W[R, col] = 1
        col += 1
        # P5 = (a+b)h
        U[a, col] = 1; U[b, col] = 1
        V[h, col] = 1
        W[P, col] = -1; W[Q, col] = 1
        col += 1
        # P6 = (c-a)(e+f)
        U[c, col] = 1; U[a, col] = -1
        V[e, col] = 1; V[f, col] = 1
        W[S, col] = 1
        col += 1
        # P7 = (b-d)(g+h)
        U[b, col] = 1; U[d, col] = -1
        V[g, col] = 1; V[h, col] = 1
        W[P, col] = 1
        col += 1

    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)

    for k_block in range(p // 2):
        for j_block in range(m // 2):
            add_strassen(1, 2 * j_block, 2 * k_block)

    return U, V, W


def _target_tensor():
    C = np.zeros((n, m, p), dtype=int)
    C[:] = 1
    return C


def _reconstruct(U, V, W):
    """Reconstruct the multiplication tensor C_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]."""
    R = U.shape[1]
    C = np.zeros((n, m, p), dtype=int)
    for r in range(R):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        if not (np.any(u_col) and np.any(v_col) and np.any(w_col)):
            continue
        # Outer product structure: u_col is (n*m,), v_col is (m*p,), w_col is (p*n,)
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    C[i, j, k] += u_val * v_col[j * p + k] * w_col[k * n + i]
    return C


def _verify(U, V, W):
    """Check exactness and return (ok, effective_rank)."""
    target = _target_tensor()
    recon = _reconstruct(U, V, W)
    ok = np.array_equal(recon, target)
    cnt = 0
    for r in range(U.shape[1]):
        if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
            cnt += 1
    return ok, cnt


def _build_contribution_matrix(U, V, W):
    """Build matrix M where M[i,j,k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    Shape: (n*m*p, R)
    """
    R = U.shape[1]
    M = np.zeros((n * m * p, R), dtype=float)
    for r in range(R):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        if not (np.any(u_col) and np.any(v_col) and np.any(w_col)):
            continue
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    idx = (i * m + j) * p + k
                    M[idx, r] = u_val * v_col[j * p + k] * w_col[k * n + i]
    return M


def _find_minimal_columns(M, target_vec, max_iter=500):
    """Find a minimal set of columns of M that sum to target_vec exactly (integer coefficients).
    
    Uses a greedy approach: start with all columns, try to remove columns that are
    linearly dependent on others with integer coefficients.
    """
    R = M.shape[1]
    target = target_vec.astype(float)
    
    # Start with all active columns
    active = list(range(R))
    
    for iteration in range(max_iter):
        if not _time_ok():
            break
        removed = False
        for idx in active[:]:
            if idx not in active:
                continue
            # Check if column idx is an integer linear combination of other active columns
            others = [c for c in active if c != idx]
            if not others:
                continue
            
            A = M[:, others]  # shape (N, len(others))
            b = M[:, idx]
            
            # Only consider rows where b is nonzero or where any column has nonzero
            mask = (b != 0) | np.any(A != 0, axis=1)
            if not np.any(mask):
                # Column is all zeros, can remove
                active.remove(idx)
                removed = True
                continue
            if np.sum(mask) < len(others):
                # Underdetermined, skip
                continue
            
            A_sub = A[mask, :]
            b_sub = b[mask]
            
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A_sub, b_sub, rcond=1e-10)
            except Exception:
                continue
            
            # Check if solution is close to integer
            if not np.allclose(sol, np.round(sol), atol=1e-8):
                continue
            
            coeffs = np.round(sol).astype(int)
            
            # Verify exactly
            recon = A_sub @ coeffs
            if not np.array_equal(recon.astype(int), b_sub.astype(int)):
                continue
            
            # Also verify on all rows
            full_recon = A @ coeffs
            if not np.array_equal(full_recon.astype(int), b.astype(int)):
                continue
            
            active.remove(idx)
            removed = True
            break
        
        if not removed:
            break
    
    return active


def _try_reduce(base_U, base_V, base_W):
    """Search for a lower-rank exact decomposition using linear algebra."""
    U = base_U.copy()
    V = base_V.copy()
    W = base_W.copy()
    R = U.shape[1]
    
    target = _target_tensor()
    target_flat = target.reshape(-1)
    
    # Build contribution matrix
    M = _build_contribution_matrix(U, V, W)
    
    # Find minimal set of columns
    active = _find_minimal_columns(M, target_flat)
    
    # Apply reduction
    U_red = np.zeros((n * m, len(active)), dtype=int)
    V_red = np.zeros((m * p, len(active)), dtype=int)
    W_red = np.zeros((p * n, len(active)), dtype=int)
    for new_idx, old_idx in enumerate(active):
        U_red[:, new_idx] = U[:, old_idx]
        V_red[:, new_idx] = V[:, old_idx]
        W_red[:, new_idx] = W[:, old_idx]
    
    # Verify
    ok, r = _verify(U_red, V_red, W_red)
    if not ok:
        return base_U.copy(), base_V.copy(), base_W.copy()
    
    # Try further reduction: for each active column, see if it can be expressed
    # as a combination of others with the constraint that the reconstruction still works
    # This is harder, so we do a simple iterative pass
    
    changed = True
    while changed and _time_ok():
        changed = False
        R_cur = U_red.shape[1]
        M_cur = _build_contribution_matrix(U_red, V_red, W_red)
        
        for idx in range(R_cur):
            if not _time_ok():
                break
            others = [c for c in range(R_cur) if c != idx]
            if not others:
                continue
            
            A = M_cur[:, others]
            b = M_cur[:, idx]
            
            mask = (b != 0) | np.any(A != 0, axis=1)
            if not np.any(mask):
                # Zero column, remove
                U_red = np.delete(U_red, idx, axis=1)
                V_red = np.delete(V_red, idx, axis=1)
                W_red = np.delete(W_red, idx, axis=1)
                changed = True
                break
            if np.sum(mask) < len(others):
                continue
            
            A_sub = A[mask, :]
            b_sub = b[mask]
            
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A_sub, b_sub, rcond=1e-10)
            except Exception:
                continue
            
            if not np.allclose(sol, np.round(sol), atol=1e-8):
                continue
            
            coeffs = np.round(sol).astype(int)
            
            full_recon = A @ coeffs
            if not np.array_equal(full_recon.astype(int), b.astype(int)):
                continue
            
            U_red = np.delete(U_red, idx, axis=1)
            V_red = np.delete(V_red, idx, axis=1)
            W_red = np.delete(W_red, idx, axis=1)
            changed = True
            break
    
    ok, r = _verify(U_red, V_red, W_red)
    if not ok:
        return base_U.copy(), base_V.copy(), base_W.copy()
    
    return U_red, V_red, W_red


def construct():
    base_U, base_V, base_W = _build_base()
    
    # Quick sanity check
    ok, r = _verify(base_U, base_V, base_W)
    if not ok:
        return {"U": base_U.tolist(), "V": base_V.tolist(), "W": base_W.tolist()}
    
    U, V, W = _try_reduce(base_U, base_V, base_W)
    
    # Final verification; if broken, fall back to base
    ok, r = _verify(U, V, W)
    if not ok:
        U, V, W = base_U, base_V, base_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
