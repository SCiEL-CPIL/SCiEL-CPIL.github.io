"""Matrix multiplication 4x4x6 via Strassen block decomposition (rank 77) + rank reduction."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]
_T_LIMIT = 115.0  # safety margin below 120s evaluator limit


def _get_strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * m + j][m_idx] += c
        for (j, k, c) in Vb:
            V[j * p + k][m_idx] += c
        for (i, k, c) in Wb:
            W[k * n + i][m_idx] += c

    return U, V, W


def _build_rank77():
    """
    Construct U, V, W for 4x4x6 matrix multiplication with rank 77.
    Structure: 49 columns for 4x4x4 (Strassen squared) + 28 columns for 4x4x2 (4 Strassen blocks).
    Returns numpy arrays.
    """
    n, m, p = 4, 4, 6
    U_s, V_s, W_s = _get_strassen_2x2_factors()
    U_s = np.array(U_s, dtype=float)
    V_s = np.array(V_s, dtype=float)
    W_s = np.array(W_s, dtype=float)

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))   # 16 x 77
    V = np.zeros((m * p, total_rank))   # 24 x 77
    W = np.zeros((p * n, total_rank))   # 24 x 77

    # --- Part 1: 4x4x4 via squared Strassen (columns 0..48) ---
    for i in range(4):
        I, i0 = divmod(i, 2)
        for j in range(4):
            J, j0 = divmod(j, 2)
            for t in range(7):
                us_t = U_s[2 * I + J][t]
                if us_t == 0:
                    continue
                for s in range(7):
                    val = us_t * U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][7 * t + s] += val

    for j in range(4):
        J, j0 = divmod(j, 2)
        for k in range(4):
            K, k0 = divmod(k, 2)
            for t in range(7):
                vs_t = V_s[2 * J + K][t]
                if vs_t == 0:
                    continue
                for s in range(7):
                    val = vs_t * V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j * 6 + k][7 * t + s] += val

    for k in range(4):
        K, k0 = divmod(k, 2)
        for i in range(4):
            I, i0 = divmod(i, 2)
            for t in range(7):
                ws_t = W_s[2 * K + I][t]
                if ws_t == 0:
                    continue
                for s in range(7):
                    val = ws_t * W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k * 4 + i][7 * t + s] += val

    # --- Part 2: 4x4x2 via 4 Strassen 2x2x2 blocks (columns 49..76) ---
    products = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
    ]

    for p_idx, (I, J, is_bot) in enumerate(products):
        col_off = rank_444 + p_idx * 7

        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                for s in range(7):
                    val = U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col_off + s] += val

        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * is_bot + j0
                k_global = 4 + k0
                for s in range(7):
                    val = V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j_global * 6 + k_global][col_off + s] += val

        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = 4 + k0
                for s in range(7):
                    val = W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k_global * 4 + i_global][col_off + s] += val

    return U, V, W


def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification. Returns max absolute error."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return float(np.max(np.abs(T - 1.0)))


def _try_reduce_rank(U0, V0, W0, n, m, p, deadline):
    """
    Vectorized greedy rank reduction:
    1. Try zeroing each column, check if error is still ~0.
    2. If a column can be removed, solve a least-squares correction on remaining columns.
    3. Iterate until no more columns can be removed or time runs out.
    """
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    R = U.shape[1]
    alive = np.ones(R, dtype=bool)
    
    # Precompute the "tensor" indices for vectorized dot products
    # For each (i,j,k): U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i
    u_idx = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    v_idx = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    w_idx = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    
    num_entries = n * m * p
    
    def compute_tensor(Um, Vm, Wm, mask=None):
        """Compute tensor values for all (i,j,k) entries."""
        if mask is not None:
            Um = Um[:, mask]
            Vm = Vm[:, mask]
            Wm = Wm[:, mask]
        # For each entry e: sum_c U[u_idx[e], c] * V[v_idx[e], c] * W[w_idx[e], c]
        # Vectorized: (num_entries, rank) @ ... no, we need element-wise product then sum
        # T[e] = sum_c U[u_idx[e],c] * V[v_idx[e],c] * W[w_idx[e],c]
        U_sel = U[u_idx, :]  # (num_entries, rank)
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        if mask is not None:
            U_sel = U_sel[:, mask]
            V_sel = V_sel[:, mask]
            W_sel = W_sel[:, mask]
        return np.sum(U_sel * V_sel * W_sel, axis=1)
    
    def check_error(Um, Vm, Wm, mask=None):
        T = compute_tensor(Um, Vm, Wm, mask)
        return float(np.max(np.abs(T - 1.0)))
    
    # Phase 1: Greedy column removal
    # Try removing each column; if error stays < tol, keep it removed.
    # Process in order of smallest column "importance" (norm).
    col_norms = (np.sum(U**2, axis=0) + np.sum(V**2, axis=0) + np.sum(W**2, axis=0)) ** 0.5
    order = np.argsort(col_norms)  # smallest first
    
    removed_count = 0
    for trial in range(R):
        if time.time() > deadline:
            break
        if not alive[trial]:
            continue
        
        # Try zeroing this column
        old_u = U[:, trial].copy()
        old_v = V[:, trial].copy()
        old_w = W[:, trial].copy()
        
        U[:, trial] = 0
        V[:, trial] = 0
        W[:, trial] = 0
        
        err = check_error(U, V, W)
        if err < 1e-8:
            # Column was redundant - keep removed
            alive[trial] = False
            removed_count += 1
        else:
            # Restore
            U[:, trial] = old_u
            V[:, trial] = old_w
            W[:, trial] = old_w
    
    # Phase 2: Try pairwise removal (more aggressive)
    # After single removals, try removing pairs
    alive_indices = np.where(alive)[0]
    for i in range(len(alive_indices)):
        if time.time() > deadline:
            break
        ci = alive_indices[i]
        if not alive[ci]:
            continue
        for j in range(i + 1, len(alive_indices)):
            if time.time() > deadline:
                break
            cj = alive_indices[j]
            if not alive[cj]:
                continue
            
            old_ui = U[:, ci].copy()
            old_vi = V[:, ci].copy()
            old_wi = W[:, ci].copy()
            old_uj = U[:, cj].copy()
            old_vj = V[:, cj].copy()
            old_wj = W[:, cj].copy()
            
            U[:, ci] = 0
            V[:, ci] = 0
            W[:, ci] = 0
            U[:, cj] = 0
            V[:, cj] = 0
            W[:, cj] = 0
            
            err = check_error(U, V, W)
            if err < 1e-8:
                alive[ci] = False
                alive[cj] = False
                removed_count += 2
            else:
                U[:, ci] = old_ui
                V[:, ci] = old_vi
                W[:, ci] = old_wi
                U[:, cj] = old_uj
                V[:, cj] = old_vj
                W[:, cj] = old_wj
                break  # If this pair doesn't work, try next i
    
    # Phase 3: Least-squares correction after removals
    # After removing columns, try to adjust remaining columns to maintain exactness
    alive_mask = alive
    if removed_count > 0 and time.time() < deadline:
        # Build the constraint system:
        # For each (i,j,k): sum_{c in alive} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is nonlinear, but we can linearize: fix V and W, solve for U (and vice versa)
        
        # Try: fix V and W, solve for each U row
        # For row u = i*m+j: sum_c U[u,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # This is a linear system in U[u, :] for each (i,j)
        
        U_new = U.copy()
        for i in range(n):
            for j in range(m):
                u_row = i * m + j
                # For each k: sum_c U[u_row, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                A_mat = np.zeros((p, R))
                for k in range(p):
                    A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Only use alive columns
                A_alive = A_mat[:, alive_mask]
                # Solve A_alive @ x = 1 (least squares)
                try:
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A_alive, np.ones(p), rcond=None)
                    # Check if solution is good
                    if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-6:
                        U_new[u_row, alive_mask] = x
                except np.linalg.LinAlgError:
                    pass
        
        # Verify the corrected U
        err = check_error(U_new, V, W)
        if err < 1e-6:
            U = U_new
        
        # Now try fixing U and W, solve for V
        V_new = V.copy()
        for j in range(m):
            for k in range(p):
                v_row = j * p + k
                # For each i: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                A_mat = np.zeros((n, R))
                for i in range(n):
                    A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                A_alive = A_mat[:, alive_mask]
                try:
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A_alive, np.ones(n), rcond=None)
                    if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-6:
                        V_new[v_row, alive_mask] = x
                except np.linalg.LinAlgError:
                    pass
        
        err = check_error(U, V_new, W)
        if err < 1e-6:
            V = V_new
        
        # Now try fixing U and V, solve for W
        W_new = W.copy()
        for k in range(p):
            for i in range(n):
                w_row = k * n + i
                # For each j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                A_mat = np.zeros((m, R))
                for j in range(m):
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                A_alive = A_mat[:, alive_mask]
                try:
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A_alive, np.ones(m), rcond=None)
                    if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-6:
                        W_new[w_row, alive_mask] = x
                except np.linalg.LinAlgError:
                    pass
        
        err = check_error(U, V, W_new)
        if err < 1e-6:
            W = W_new
    
    # Final verification and compaction
    final_err = check_error(U, V, W)
    if final_err > 1e-6:
        # Fallback: return original
        return U0, V0, W0
    
    # Compact: remove dead columns
    keep = np.where(alive)[0]
    final_rank = len(keep)
    U_final = U[:, keep]
    V_final = V[:, keep]
    W_final = W[:, keep]
    
    # Round to nearest integer (entries should be integers or half-integers)
    U_final = np.round(U_final)
    V_final = np.round(V_final)
    W_final = np.round(W_final)
    
    # Verify after rounding
    err_rounded = check_error(U_final, V_final, W_final)
    if err_rounded < 1e-6:
        return U_final, V_final, W_final
    else:
        # Try rounding to half-integers
        U_half = np.round(U_final * 2) / 2.0
        V_half = np.round(V_final * 2) / 2.0
        W_half = np.round(W_final * 2) / 2.0
        err_half = check_error(U_half, V_half, W_half)
        if err_half < 1e-6:
            return U_half, V_half, W_half
        else:
            return U0, V0, W0


def _fallback_rank96():
    """Trivial rank-96 decomposition (one column per (i,j,k) triple)."""
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _to_list(mat):
    """Convert numpy array to list of lists of numbers."""
    return mat.tolist()


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + _T_LIMIT

    # Build the rank-77 structured decomposition
    U, V, W = _build_rank77()
    
    # Verify base decomposition
    err = _verify_fast(U, V, W, n, m, p)
    if err > 1e-6:
        # Something went wrong with construction; use fallback
        U, V, W = _fallback_rank96()
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    # Try to reduce rank
    if time.time() < deadline - 5:
        U2, V2, W2 = _try_reduce_rank(U, V, W, n, m, p, deadline)
        if U2.shape[1] < U.shape[1]:
            U, V, W = U2, V2, W2
    
    # Final verification
    final_err = _verify_fast(U, V, W, n, m, p)
    if final_err > 1e-6:
        U, V, W = _fallback_rank96()
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
