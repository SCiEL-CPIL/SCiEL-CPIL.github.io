import numpy as np
import time

def construct():
    """
    Rank-reduced Strassen-based decomposition for 4x4 * 4x8 matrix multiplication.
    
    Builds a rank-98 decomposition via Kronecker product of Strassen 2x2x2,
    then checks for linear dependencies in the tensor columns.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition
    result = construct_rank98(n, m, p)
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        return construct_naive(n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank via direct tensor column rank check
    rank_reduced = _reduce_rank_direct(U, V, W, n, m, p, deadline - start_time)
    
    if rank_reduced is not None:
        U_r, V_r, W_r = rank_reduced
        if _verify_decomposition(U_r, V_r, W_r, n, m, p):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Try simple column elimination: remove columns that are zero in all three factors
    U_clean, V_clean, W_clean = _eliminate_zero_columns(U, V, W)
    if U_clean.shape[1] < U.shape[1]:
        if _verify_decomposition(U_clean, V_clean, W_clean, n, m, p):
            return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
    
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


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Test with random matrices
    rng = np.random.default_rng(42)
    for trial in range(3):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        M = np.zeros(rank)
        for t in range(rank):
            u_val = np.dot(U[:, t], A_flat)
            v_val = np.dot(V[:, t], B_flat)
            M[t] = u_val * v_val
        
        C_computed = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                C_computed[i, k] = np.dot(W[k*n+i, :], M)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _reduce_rank_direct(U, V, W, n, m, p, time_budget):
    """
    Direct tensor column rank check: flatten each tensor column and check
    linear independence. If rank < number of columns, attempt reduction.
    """
    rank = U.shape[1]
    dim = n * m * p
    elapsed_start = time.time()
    
    # Build the tensor column matrix: each column t is the vector
    # T_t[i,j,k] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # Flatten in order (i,j,k) -> index i*m*p + j*p + k
    
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
    
    if time.time() - elapsed_start > time_budget * 0.3:
        return None
    
    # Check rank of T (should be <= min(dim, rank) = min(128, 98) = 98)
    try:
        s = np.linalg.svd(T, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    
    tol = 1e-10
    num_rank = int(np.sum(s > tol))
    
    if num_rank >= rank:
        # No reduction possible - all 98 columns are linearly independent
        return None
    
    # If num_rank < rank, attempt to find a valid tensor rank reduction
    # by finding linear dependencies and absorbing them into the factors
    
    # Get the column space basis from SVD
    U_svd, s_full, Vt = np.linalg.svd(T, full_matrices=False)
    
    # Find independent columns
    independent = []
    for t in range(rank):
        col = T[:, t]
        if not independent:
            independent.append(t)
        else:
            # Check if col is in span of T[:, independent]
            A = T[:, independent]
            try:
                coeffs, residuals, _, _ = np.linalg.lstsq(A, col, rcond=1e-12)
                if residuals.size > 0 and residuals[0] > tol:
                    independent.append(t)
            except np.linalg.LinAlgError:
                independent.append(t)
    
    num_independent = len(independent)
    if num_independent >= rank:
        return None
    
    # Build new decomposition using only independent columns
    # This is a heuristic - it may not produce a valid tensor decomposition
    # but let's try and verify
    
    U_new = U[:, independent]
    V_new = V[:, independent]
    W_new = W[:, independent]
    
    return U_new, V_new, W_new


def _greedy_basis_selection(coords, num_rank):
    """Greedy selection of linearly independent rows from coords."""
    rank = coords.shape[0]
    selected = []
    
    for t in range(rank):
        if len(selected) >= num_rank:
            break
        if not selected:
            selected.append(t)
        else:
            A = coords[selected, :]
            b = coords[t, :]
            try:
                x, residuals, _, _ = np.linalg.lstsq(A.T, b, rcond=1e-12)
                if residuals.size > 0 and residuals[0] < 1e-10:
                    continue
            except np.linalg.LinAlgError:
                continue
            selected.append(t)
    
    if len(selected) < num_rank:
        return None
    return selected[:num_rank]


def _eliminate_zero_columns(U, V, W):
    """Remove columns that are zero in any of U, V, W."""
    rank = U.shape[1]
    keep = []
    for t in range(rank):
        if np.all(np.abs(U[:, t]) > 1e-15) and np.all(np.abs(V[:, t]) > 1e-15) and np.all(np.abs(W[:, t]) > 1e-15):
            keep.append(t)
    
    if len(keep) == rank:
        return U, V, W
    
    return U[:, keep], V[:, keep], W[:, keep]
