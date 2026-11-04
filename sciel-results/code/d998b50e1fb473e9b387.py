import numpy as np
import time

def construct():
    """
    Rank-reduced decomposition for 4x4 * 4x8 matrix multiplication.
    
    Builds a rank-98 decomposition via Kronecker product of Strassen 2x2x2,
    then attempts rank reduction via tensor column linear dependencies.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition
    result = construct_rank98(n, m, p)
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    # Verify the base decomposition
    try:
        if not _verify_decomposition_fast(U, V, W, n, m, p):
            return construct_naive(n, m, p)
    except Exception:
        return construct_naive(n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > 80:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank via direct tensor column rank check
    try:
        rank_reduced = _reduce_rank_direct(U, V, W, n, m, p, deadline - start_time)
        if rank_reduced is not None:
            U_r, V_r, W_r = rank_reduced
            if _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    except Exception:
        pass
    
    # Try additional reduction: exploit rectangular structure
    try:
        rank_reduced2 = _reduce_rank_rectangular(U, V, W, n, m, p, deadline - start_time)
        if rank_reduced2 is not None:
            U_r, V_r, W_r = rank_reduced2
            if _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    except Exception:
        pass
    
    # Return the base decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct_rank98(n, m, p):
    """Rank-98 decomposition using Kronecker product of Strassen levels."""
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block (output cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using matrix operations."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Test with random matrices using matrix operations
    rng = np.random.default_rng(42)
    for trial in range(3):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        # Vectorized computation
        M = (U @ A_flat) * (V @ B_flat)
        
        C_flat = W @ M
        C_computed = C_flat.reshape((n, p), order='F')
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    return _verify_decomposition_fast(U, V, W, n, m, p)


def _reduce_rank_direct(U, V, W, n, m, p, time_budget):
    """
    Direct tensor column rank check: flatten each tensor column and check
    linear independence. If rank < number of columns, attempt reduction.
    """
    rank = U.shape[1]
    dim = n * m * p
    elapsed_start = time.time()
    
    # Build index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_flat_idx = i_idx * m + j_idx
    v_flat_idx = j_idx * p + k_idx
    w_flat_idx = k_idx * n + i_idx
    
    U_vals = U[u_flat_idx, :]  # (dim, rank)
    V_vals = V[v_flat_idx, :]  # (dim, rank)
    W_vals = W[w_flat_idx, :]  # (dim, rank)
    
    # Tensor columns: T[:, t] = U_vals[:, t] * V_vals[:, t] * W_vals[:, t]
    T = U_vals * V_vals * W_vals  # (dim, rank)
    
    if time.time() - elapsed_start > time_budget * 0.2:
        return None
    
    # Check rank of T
    try:
        s = np.linalg.svd(T, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    
    tol = 1e-10
    num_rank = int(np.sum(s > tol))
    
    if num_rank >= rank:
        return None
    
    # Find which columns are dependent and express them in terms of independent ones
    # Use QR with column pivoting to identify independent columns
    try:
        Q, R, piv = _qr_pivot(T, num_rank)
    except np.linalg.LinAlgError:
        return None
    
    if time.time() - elapsed_start > time_budget * 0.4:
        return None
    
    # piv[0:num_rank] are the independent columns
    # piv[num_rank:] are the dependent columns
    independent_cols = sorted(piv[:num_rank].tolist())
    dependent_cols = sorted(piv[num_rank:].tolist())
    
    # For each dependent column, find its representation in terms of independent ones
    T_indep = T[:, independent_cols]
    T_dep = T[:, dependent_cols]
    
    try:
        coeffs, residuals, rank_coeffs, _ = np.linalg.lstsq(T_indep, T_dep, rcond=1e-12)
    except np.linalg.LinAlgError:
        return None
    
    # Verify the representation is accurate
    if not np.allclose(T_indep @ coeffs, T_dep, atol=1e-8):
        return None
    
    # Build the modified W
    W_mod = W[:, independent_cols].copy()
    for d_idx, dep_col in enumerate(dependent_cols):
        for j_idx, indep_col in enumerate(independent_cols):
            c = coeffs[j_idx, d_idx]
            if abs(c) > 1e-14:
                W_mod[:, j_idx] += c * W[:, dep_col]
    
    U_reduced = U[:, independent_cols]
    V_reduced = V[:, independent_cols]
    
    if time.time() - elapsed_start > time_budget * 0.6:
        return None
    
    if _verify_decomposition_fast(U_reduced, V_reduced, W_mod, n, m, p):
        return (U_reduced, V_reduced, W_mod)
    
    # Try the reverse: modify V factors instead
    V_mod = V[:, independent_cols].copy()
    W_reduced = W[:, independent_cols].copy()
    for d_idx, dep_col in enumerate(dependent_cols):
        for j_idx, indep_col in enumerate(independent_cols):
            c = coeffs[j_idx, d_idx]
            if abs(c) > 1e-14:
                V_mod[:, j_idx] += c * V[:, dep_col]
    
    if _verify_decomposition_fast(U_reduced, V_mod, W_reduced, n, m, p):
        return (U_reduced, V_mod, W_reduced)
    
    # Try modifying U factors instead
    U_mod = U[:, independent_cols].copy()
    V_reduced = V[:, independent_cols].copy()
    W_reduced = W[:, independent_cols].copy()
    for d_idx, dep_col in enumerate(dependent_cols):
        for j_idx, indep_col in enumerate(independent_cols):
            c = coeffs[j_idx, d_idx]
            if abs(c) > 1e-14:
                U_mod[:, j_idx] += c * U[:, dep_col]
    
    if _verify_decomposition_fast(U_mod, V_reduced, W_reduced, n, m, p):
        return (U_mod, V_reduced, W_reduced)
    
    # Try iterative removal of columns with smallest SVD singular values
    if time.time() - elapsed_start > time_budget * 0.8:
        return None
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    active_cols = list(range(rank))
    
    # Sort columns by their contribution (use norm of tensor column)
    col_norms = np.linalg.norm(T, axis=0)
    col_order = np.argsort(col_norms)
    
    for col_to_remove in col_order:
        if time.time() - elapsed_start > time_budget * 0.9:
            break
        if len(active_cols) <= num_rank:
            break
            
        # Try removing this column
        new_active = [c for c in active_cols if c != col_to_remove]
        if len(new_active) < 1:
            break
        
        U_test = U[:, new_active]
        V_test = V[:, new_active]
        W_test = W[:, new_active]
        
        if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p):
            active_cols = new_active
            U_cur = U_test
            V_cur = V_test
            W_cur = W_test
    
    if len(active_cols) < rank:
        return (U_cur, V_cur, W_cur)
    
    return None


def _reduce_rank_rectangular(U, V, W, n, m, p, time_budget):
    """
    Exploit the rectangular structure: the 4x4x8 tensor can be seen as two
    4x4x4 blocks sharing the same A-factor. Try to merge or reduce across blocks.
    """
    rank = U.shape[1]
    elapsed_start = time.time()
    
    # The structure is: U = [U4, U4], V = [V4_left, V4_right], W = [W4_top, W4_bottom]
    # where V4_left uses B's first 4 cols, V4_right uses B's last 4 cols
    # W4_top uses output's first 4 rows, W4_bottom uses output's last 4 rows
    
    # Try: can we find a lower-rank representation by exploiting that
    # both blocks use the same U4?
    # This means the A-projections are duplicated.
    
    # Approach: try to find if columns in the second block are linear combinations
    # of columns in the first block when considering the full tensor structure.
    
    dim = n * m * p
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_flat_idx = i_idx * m + j_idx
    v_flat_idx = j_idx * p + k_idx
    w_flat_idx = k_idx * n + i_idx
    
    U_vals = U[u_flat_idx, :]
    V_vals = V[v_flat_idx, :]
    W_vals = W[w_flat_idx, :]
    
    T = U_vals * V_vals * W_vals
    
    # The first 49 columns correspond to output cols 0-3
    # The last 49 columns correspond to output cols 4-7
    
    # Check if any column in the second half is a linear combination of the first half
    # (this would indicate redundancy)
    T_first = T[:, 0:49]
    T_second = T[:, 49:98]
    
    try:
        # Check if T_second columns are in the span of T_first
        Q, R, piv = _qr_pivot(T_first, 49)
        rank_first = int(np.sum(np.abs(np.diag(R)) > 1e-10))
        
        if rank_first < 49:
            # First block has internal dependencies - handle separately
            pass
        
        # Try to express T_second as T_first @ M
        coeffs, residuals, _, _ = np.linalg.lstsq(T_first, T_second, rcond=1e-12)
        
        if np.allclose(T_first @ coeffs, T_second, atol=1e-8):
            # T_second = T_first @ coeffs
            # This means we can potentially merge
            # But we need to check if this leads to a valid reduced decomposition
            # For now, just note the relationship
            pass
    except np.linalg.LinAlgError:
        pass
    
    if time.time() - elapsed_start > time_budget * 0.3:
        return None
    
    # Alternative: try greedy column elimination with verification
    # Start from the full 98 columns and try to remove one at a time
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    active = list(range(rank))
    
    # Sort by tensor column norm (smallest first)
    col_norms = np.linalg.norm(T, axis=0)
    col_order = np.argsort(col_norms)
    
    for col in col_order:
        if time.time() - elapsed_start > time_budget * 0.95:
            break
        if len(active) <= 49:  # Don't go below one block's worth
            break
        
        new_active = [c for c in active if c != col]
        if not new_active:
            break
        
        U_test = U[:, new_active]
        V_test = V[:, new_active]
        W_test = W[:, new_active]
        
        if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p):
            active = new_active
            U_cur = U_test
            V_cur = V_test
            W_cur = W_test
    
    if len(active) < rank:
        return (U_cur, V_cur, W_cur)
    
    return None


def _qr_pivot(A, num_rank):
    """QR decomposition with column pivoting to find independent columns."""
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    piv = list(range(n))
    
    for k in range(min(m, n)):
        # Find column with largest norm in remaining part
        if k < n:
            norms = np.linalg.norm(R[k:, k:], axis=0)
            max_col = np.argmax(norms) + k
            if max_col != k:
                R[:, [k, max_col]] = R[:, [max_col, k]]
                piv[k], piv[max_col] = piv[max_col], piv[k]
        
        # QR step
        col_norm = np.linalg.norm(R[k:, k])
        if col_norm < 1e-14:
            break
        v = R[k:, k] / col_norm
        R[k:, k:] -= col_norm * np.outer(v, R[k:, k:] @ v)
        R[k, k] = col_norm
        Q[k:, :] -= col_norm * np.outer(v, Q[k:, :] @ v)
    
    return Q, R, piv
