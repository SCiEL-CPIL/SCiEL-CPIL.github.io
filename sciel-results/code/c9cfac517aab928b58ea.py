"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _compute_tensor(U, V, W, n, m, p):
    """Compute the full tensor T[i,j,k] from the decomposition."""
    T = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return T

def _try_remove_term(U, V, W, idx, n, m, p):
    """
    Try to remove term idx by seeing if the remaining terms can be adjusted.
    This is a simple check: if the term is already zero, remove it.
    """
    if np.allclose(U[:, idx], 0) or np.allclose(V[:, idx], 0) or np.allclose(W[:, idx], 0):
        return True, idx
    return False, -1

def _try_merge_terms(U, V, W, idx1, idx2, n, m, p):
    """
    Try to merge terms idx1 and idx2 into a single rank-1 term.
    The combined tensor T12 = T1 + T2 must be rank-1.
    If so, find the rank-1 decomposition.
    Returns (success, new_U_col, new_V_col, new_W_col) or (False, None, None, None).
    """
    T1 = np.einsum('i,j,k->ijk', U[:, idx1], V[:, idx1], W[:, idx1])
    T2 = np.einsum('i,j,k->ijk', U[:, idx2], V[:, idx2], W[:, idx2])
    T12 = T1 + T2
    
    # Check if T12 is rank-1 by checking all 2x2 minors of any matricization
    # Matricize as (n) x (m*p)
    T12_mat = T12.reshape(n, m * p)
    
    # Check rank-1: all 2x2 minors must be zero
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for j1 in range(m * p):
                for j2 in range(j1 + 1, m * p):
                    det = T12_mat[i1, j1] * T12_mat[i2, j2] - T12_mat[i1, j2] * T12_mat[i2, j1]
                    if abs(det) > 1e-9:
                        return False, None, None, None
    
    # T12 is rank-1 (or zero). If zero, we can just drop both terms.
    if np.allclose(T12, 0):
        return True, None, None, None  # Special case: both terms cancel
    
    # Find the rank-1 decomposition of T12
    # T12[i,j,k] = u[i] * v[j] * w[k]
    # Pick a non-zero entry to normalize
    u = np.zeros(n)
    v = np.zeros(m)
    w = np.zeros(p)
    
    # Find first non-zero entry
    found = False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T12[i, j, k]) > 1e-12:
                    u[i] = 1.0
                    v[j] = 1.0
                    w[k] = T12[i, j, k]
                    found = True
                    break
            if found:
                break
        if found:
            break
    
    if not found:
        return True, None, None, None  # Zero tensor
    
    # Now fill in the rest using the rank-1 structure
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == 0 and j == 0 and k == 0:
                    continue
                # T12[i,j,k] = u[i]*v[j]*w[k]
                # We need to determine u[i], v[j], w[k] consistently
                pass
    
    # Better approach: use SVD on the matricization
    # T12_mat = u_vec * v_vec^T where u_vec has length n, v_vec has length m*p
    S, _, _ = np.linalg.svd(T12_mat, full_matrices=False)
    if len(S) > 0 and S[0] > 1e-12:
        u_vec = S[0] * np.linalg.svd(T12_mat, full_matrices=False)[0][:, 0]
        v_vec = S[0] * np.linalg.svd(T12_mat, full_matrices=False)[2][0, :]
        # v_vec is length m*p, reshape to (m, p)
        v_mat = v_vec.reshape(m, p)
        # Now T12[i,j,k] = u_vec[i] * v_mat[j,k]
        # But v_mat[j,k] must be v[j]*w[k] for rank-1 in the 3-way sense
        # Check if v_mat is rank-1
        v_mat_S, _, _ = np.linalg.svd(v_mat, full_matrices=False)
        if len(v_mat_S) > 1 and v_mat_S[1] > 1e-9 * v_mat_S[0]:
            return False, None, None, None  # Not separable in j,k
        
        # v_mat[j,k] = v[j] * w[k]
        # Find v and w
        v = np.zeros(m)
        w = np.zeros(p)
        # Normalize using first non-zero row/col
        for j in range(m):
            for k in range(p):
                if abs(v_mat[j, k]) > 1e-12:
                    if abs(v[j]) < 1e-12:
                        v[j] = 1.0
                        w[k] = v_mat[j, k]
                    elif abs(w[k]) < 1e-12:
                        w[k] = v_mat[j, k] / v[j]
                    break
            if abs(v[j]) > 1e-12:
                continue
            # Row j is all zeros, v[j] = 0
            v[j] = 0
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(u_vec[i] * v[j] * w[k] - T12[i, j, k]) > 1e-9:
                        return False, None, None, None
        
        return True, u_vec, v, w
    
    return False, None, None, None

def _rank_reduce_search(U, V, W, n, m, p, time_limit):
    """
    Iteratively try to reduce rank by merging or removing terms.
    """
    start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Precompute tensor for verification
    T_target = _compute_tensor(U, V, W, n, m, p)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        r = U.shape[1]
        
        # Try removing individual terms
        for idx in range(r):
            if time.time() - start_time > time_limit:
                break
            success, _ = _try_remove_term(U, V, W, idx, n, m, p)
            if success:
                U = np.delete(U, idx, axis=1)
                V = np.delete(V, idx, axis=1)
                W = np.delete(W, idx, axis=1)
                r -= 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try merging pairs of terms
        for idx1 in range(r):
            if time.time() - start_time > time_limit:
                break
            for idx2 in range(idx1 + 1, r):
                if time.time() - start_time > time_limit:
                    break
                success, u_new, v_new, w_new = _try_merge_terms(U, V, W, idx1, idx2, n, m, p)
                if success:
                    if u_new is not None:
                        # Replace terms idx1 and idx2 with a single new term
                        # Remove idx2 first (higher index), then idx1
                        U = np.delete(U, idx2, axis=1)
                        V = np.delete(V, idx2, axis=1)
                        W = np.delete(W, idx2, axis=1)
                        U = np.delete(U, idx1, axis=1)
                        V = np.delete(V, idx1, axis=1)
                        W = np.delete(W, idx1, axis=1)
                        
                        # Add new term
                        U = np.column_stack([U, u_new])
                        V = np.column_stack([V, v_new])
                        W = np.column_stack([W, w_new])
                    else:
                        # Both terms cancel, just remove both
                        U = np.delete(U, idx2, axis=1)
                        V = np.delete(V, idx2, axis=1)
                        W = np.delete(W, idx2, axis=1)
                        U = np.delete(U, idx1, axis=1)
                        V = np.delete(V, idx1, axis=1)
                        W = np.delete(W, idx1, axis=1)
                    
                    r = U.shape[1]
                    improved = True
                    break
            if improved:
                break
    
    return U, V, W

def _construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Should not happen, but fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to reduce rank
    U_reduced, V_reduced, W_reduced = _rank_reduce_search(U, V, W, n, m, p, 100)
    
    # Verify reduced version
    if _verify(U_reduced.tolist(), V_reduced.tolist(), W_reduced.tolist(), n, m, p):
        return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
    
    # Fallback to schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    return _construct()
# EVOLVE-BLOCK-END
