import numpy as np
import time

def construct():
    """
    Rank-reduced Strassen-based decomposition for 4x4 * 4x8 matrix multiplication.
    
    Builds a rank-98 decomposition via Kronecker product of Strassen 2x2x2,
    then searches for linearly dependent columns to reduce the rank.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition
    result = construct_rank98(n, m, p)
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to naive if verification fails
        return construct_naive(n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > 120:
        # Not enough time for reduction, return base
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linearly dependent columns
    rank_reduced = _reduce_rank(U, V, W, n, m, p, deadline - start_time)
    
    if rank_reduced is not None:
        U_r, V_r, W_r = rank_reduced
        if _verify_decomposition(U_r, V_r, W_r, n, m, p):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
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
    for trial in range(5):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Compute using decomposition
        # M_t = sum_{i,j} U[i,j,t] * A[i,j] * (sum_{k} V[j,k,t] * B[j,k])
        # Actually: M_t = (U[:,t]^T * A_flat) * (V[:,t]^T * B_flat) where A_flat is row-major
        # C[i,k] = sum_t W[k,i,t] * M_t
        
        A_flat = A.flatten()  # n*m
        B_flat = B.flatten()  # m*p
        
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

def _reduce_rank(U, V, W, n, m, p, time_budget):
    """
    Try to reduce the rank by finding linearly dependent column tensors.
    If T_t = sum_s alpha_ts T_s, we can remove t and scale s by (1 + sum_t alpha_ts).
    Uses vectorized tensor construction and iterative elimination.
    """
    rank = U.shape[1]
    dim = n * m * p
    elapsed_start = time.time()
    
    # Vectorized construction of all column tensors
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each t, T_t[i,j,k] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Reshape for broadcasting:
    # U reshaped to (n, m, rank)
    U_r = U.reshape(n, m, rank)
    # V reshaped to (m, p, rank)
    V_r = V.reshape(m, p, rank)
    # W reshaped to (p, n, rank)
    W_r = W.reshape(p, n, rank)
    
    # For each (i,j,k), T_t[i,j,k] = U_r[i,j,t] * V_r[j,k,t] * W_r[k,i,t]
    # We need to sum over t to form the full tensor, but we want individual columns.
    # Instead, build col_tensors[t] = flatten(T_t)
    
    # Efficient approach: 
    # col_tensors[t, i*m+j*p? no, i*m+j is index for U, but T is indexed by (i,j,k)
    # T[i,j,k] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # Flatten index: idx = i*m*p + j*p + k
    
    col_tensors = np.zeros((rank, dim))
    
    # Precompute indices for U, V, W lookups
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # For each t, we need:
    # col_tensors[t, idx] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # where idx = i*m*p + j*p + k
    
    # Vectorized:
    # U_vals: (dim, rank) where U_vals[idx, t] = U[i*m+j, t]
    # V_vals: (dim, rank) where V_vals[idx, t] = V[j*p+k, t]
    # W_vals: (dim, rank) where W_vals[idx, t] = W[k*n+i, t]
    
    u_flat_idx = i_idx * m + j_idx  # shape (dim,)
    v_flat_idx = j_idx * p + k_idx  # shape (dim,)
    w_flat_idx = k_idx * n + i_idx  # shape (dim,)
    
    # U_vals[idx, t] = U[u_flat_idx[idx], t]
    # This is a gather operation
    U_vals = U[u_flat_idx, :]  # (dim, rank)
    V_vals = V[v_flat_idx, :]  # (dim, rank)
    W_vals = W[w_flat_idx, :]  # (dim, rank)
    
    # col_tensors[t, idx] = U_vals[idx, t] * V_vals[idx, t] * W_vals[idx, t]
    # So col_tensors = (U_vals * V_vals * W_vals).T  # (rank, dim)
    col_tensors = (U_vals * V_vals * W_vals).T  # (rank, dim)
    
    # Now find linear dependencies
    # Use iterative elimination with lstsq
    kept_indices = []
    coefficients = {} # t -> {s: alpha}
    
    for t in range(rank):
        if time.time() - elapsed_start > time_budget * 0.9:
            break
        
        if not kept_indices:
            kept_indices.append(t)
        else:
            # Check if col_tensors[t] is in the span of col_tensors[:, kept_indices]
            A = col_tensors[:, kept_indices].T  # (dim, k)
            b = col_tensors[t]
            try:
                x, residuals, rank_mat, sv = np.linalg.lstsq(A, b, rcond=1e-12)
                if residuals.size > 0 and residuals[0] < 1e-12:
                    # Dependent
                    coeff_dict = {}
                    for idx, s in enumerate(kept_indices):
                        if abs(x[idx]) > 1e-12:
                            coeff_dict[s] = x[idx]
                    coefficients[t] = coeff_dict
                else:
                    kept_indices.append(t)
            except np.linalg.LinAlgError:
                kept_indices.append(t)
    
    if len(kept_indices) < rank:
        # Calculate scaling factors for kept columns
        scale = np.ones(len(kept_indices))
        # Map kept_indices to indices in scale array
        kept_to_idx = {s: idx for idx, s in enumerate(kept_indices)}
        
        for t, coeffs in coefficients.items():
            for s, alpha in coeffs.items():
                if s in kept_to_idx:
                    scale[kept_to_idx[s]] += alpha
        
        # Apply scaling to U
        U_reduced = U[:, kept_indices].copy()
        for idx, s in enumerate(kept_indices):
            if abs(scale[idx]) > 1e-12:
                U_reduced[:, idx] *= scale[idx]
        
        V_reduced = V[:, kept_indices].copy()
        W_reduced = W[:, kept_indices].copy()
        
        if _verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p):
            return U_reduced, V_reduced, W_reduced
    
    return None
