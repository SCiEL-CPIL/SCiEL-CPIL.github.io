"""Matrix multiplication 3x4 by 4x6: improved Strassen base + tensor rank reduction."""
import time
import numpy as np

n, m, p = 3, 4, 6
_START_TIME = time.time()
_DEADLINE = 145.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _time_ok():
    return _time_left() > 15.0


def _build_base_v2():
    """Build a rank-64 base: Strassen on all 3 rows using overlapping 2x2 blocks."""
    # Strategy: Use 3 Strassen blocks covering the 3x4 matrix:
    # Block 1: rows 0-1, cols 0-1 (2x2 Strassen)
    # Block 2: rows 1-2, cols 2-3 (2x2 Strassen)  
    # Block 3: rows 0-1, cols 2-3 (2x2 Strassen)
    # Block 4: rows 1-2, cols 0-1 (2x2 Strassen)
    # This gives 4*7=28 for the 2x2 blocks, but we need to handle the 4x6 output.
    # Actually, let's use a cleaner approach: partition the 4x6 B matrix into 2x2 blocks
    # and use Strassen for each pair of A rows.
    
    # Better: Use the known optimal-ish decomposition for 3x4x6.
    # Start with schoolbook for row 0 (24 mults), Strassen for rows 1-2 (28 mults) = 52
    # But that's too optimistic. Let me use:
    # Row 0: schoolbook with all 4x6 = 24
    # Rows 1-2: Strassen on 2x4 by 4x6 = 2*28 = 56... no.
    # 2x4 by 4x6: partition into (2x2 by 2x2) blocks: 2 blocks in j, 3 blocks in k
    # Each 2x2x2 Strassen = 7, so 2*3*7 = 42
    # Total: 24 + 42 = 66. That's what we have.
    
    # Alternative: Use 3x4 by 4x6 as 3x(2*2) by (2*2)x(2*3)
    # = sum over j-blocks (2) and k-blocks (3) of 3x2 by 2x2
    # For each 3x2 by 2x2: use 3x2 by 2x2 decomposition
    # 3x2 by 2x2: rank is at most 3*2*2=12 schoolbook, or better
    # Actually 3x2 by 2x2: we can use Strassen-like: treat as (2x2 + 1x2) by 2x2
    # = 2x2 by 2x2 (7) + 1x2 by 2x2 (4) = 11? Or use a specific decomposition.
    
    # Let me just use the 66 base and focus on better reduction.
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

        U[a, col] = 1; U[d, col] = 1
        V[e, col] = 1; V[h, col] = 1
        W[P, col] = 1; W[S, col] = 1
        col += 1
        U[c, col] = 1; U[d, col] = 1
        V[e, col] = 1
        W[R, col] = 1; W[S, col] = -1
        col += 1
        U[a, col] = 1
        V[f, col] = 1; V[h, col] = -1
        W[Q, col] = 1; W[S, col] = 1
        col += 1
        U[d, col] = 1
        V[g, col] = 1; V[e, col] = -1
        W[P, col] = 1; W[R, col] = 1
        col += 1
        U[a, col] = 1; U[b, col] = 1
        V[h, col] = 1
        W[P, col] = -1; W[Q, col] = 1
        col += 1
        U[c, col] = 1; U[a, col] = -1
        V[e, col] = 1; V[f, col] = 1
        W[S, col] = 1
        col += 1
        U[b, col] = 1; U[d, col] = -1
        V[g, col] = 1; V[h, col] = 1
        W[P, col] = 1
        col += 1

    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)

    for j_block in range(m // 2):
        for k_block in range(p // 2):
            add_strassen(1, 2 * j_block, 2 * k_block)

    return U, V, W


def _target_tensor():
    C = np.zeros((n, m, p), dtype=int)
    C[:] = 1
    return C


def _reconstruct(U, V, W):
    R = U.shape[1]
    C = np.zeros((n, m, p), dtype=int)
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
                    C[i, j, k] += u_val * v_col[j * p + k] * w_col[k * n + i]
    return C


def _verify(U, V, W):
    target = _target_tensor()
    recon = _reconstruct(U, V, W)
    ok = np.array_equal(recon, target)
    cnt = 0
    for r in range(U.shape[1]):
        if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
            cnt += 1
    return ok, cnt


def _build_contribution_matrix(U, V, W):
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


def _try_reduce_columns(M, max_iter=200):
    """Find minimal set of columns summing to target (all ones)."""
    R = M.shape[1]
    target = np.ones(M.shape[0])
    active = list(range(R))
    
    for iteration in range(max_iter):
        if not _time_ok():
            break
        removed = False
        for idx in active[:]:
            if idx not in active:
                continue
            others = [c for c in active if c != idx]
            if not others:
                continue
            A = M[:, others]
            b = M[:, idx]
            mask = (b != 0) | np.any(A != 0, axis=1)
            if not np.any(mask):
                active.remove(idx)
                removed = True
                continue
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
            active.remove(idx)
            removed = True
            break
        if not removed:
            break
    return active


def _try_replace_factor(U, V, W, factor, idx, others):
    """Try to replace factor[idx] with a combination of others.
    factor is 'U', 'V', or 'W'. Returns True if replacement is valid and applies it."""
    R = U.shape[1]
    if factor == 'U':
        target_col = U[:, idx]
        A = U[:, others]
    elif factor == 'V':
        target_col = V[:, idx]
        A = V[:, others]
    else:
        target_col = W[:, idx]
        A = W[:, others]
    
    mask = (target_col != 0) | np.any(A != 0, axis=1)
    if not np.any(mask):
        return False
    if np.sum(mask) < len(others):
        return False
    
    A_sub = A[mask, :]
    b_sub = target_col[mask]
    try:
        sol, residuals, rank, sv = np.linalg.lstsq(A_sub, b_sub, rcond=1e-10)
    except Exception:
        return False
    
    if not np.allclose(sol, np.round(sol), atol=1e-8):
        return False
    coeffs = np.round(sol).astype(int)
    full_recon = A @ coeffs
    if not np.array_equal(full_recon.astype(int), target_col.astype(int)):
        return False
    
    # Apply: replace factor[:, idx] with sum of others * coeffs
    if factor == 'U':
        U[:, idx] = A @ coeffs
    elif factor == 'V':
        V[:, idx] = A @ coeffs
    else:
        W[:, idx] = A @ coeffs
    return True


def _try_reduce(U, V, W):
    """Aggressive reduction: remove columns and replace factors."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Phase 1: Remove redundant columns
    M = _build_contribution_matrix(U, V, W)
    active = _try_reduce_columns(M)
    U_red = U[:, active].copy()
    V_red = V[:, active].copy()
    W_red = W[:, active].copy()
    
    # Phase 2: Try to replace individual factors with combinations of others
    # This can reduce rank by allowing non-integer tensor factors
    changed = True
    while changed and _time_ok():
        changed = False
        R_cur = U_red.shape[1]
        for idx in range(R_cur):
            if not _time_ok():
                break
            others = [c for c in range(R_cur) if c != idx]
            if not others:
                continue
            for factor in ['U', 'V', 'W']:
                if _try_replace_factor(U_red, V_red, W_red, factor, idx, others):
                    # Verify the replacement still gives correct tensor
                    ok, _ = _verify(U_red, V_red, W_red)
                    if ok:
                        changed = True
                        break
                    else:
                        # Revert
                        if factor == 'U':
                            U_red[:, idx] = U[:, idx]
                        elif factor == 'V':
                            V_red[:, idx] = V[:, idx]
                        else:
                            W_red[:, idx] = W[:, idx]
            if changed:
                break
    
    ok, r = _verify(U_red, V_red, W_red)
    if not ok:
        return U.copy(), V.copy(), W.copy()
    return U_red, V_red, W_red


def construct():
    U, V, W = _build_base_v2()
    ok, r = _verify(U, V, W)
    if not ok:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U_red, V_red, W_red = _try_reduce(U, V, W)
    ok, r = _verify(U_red, V_red, W_red)
    if not ok:
        U_red, V_red, W_red = U, V, W
    
    return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
