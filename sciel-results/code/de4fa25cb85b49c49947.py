"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    nm = n * m  # 9
    mp = m * p  # 21
    np_ = n * p  # 21
    
    # Build the target tensor entries
    # T[i*m+j, j*p+k, k*n+i] = 1
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                tensor_entries.append((idx_a, idx_b, idx_c))
                
    A_idx = np.array([e[0] for e in tensor_entries])
    B_idx = np.array([e[1] for e in tensor_entries])
    C_idx = np.array([e[2] for e in tensor_entries])
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound is max(nm, mp, np_) = 21
    # Known upper bounds suggest trying from 30 upwards
    best_rank = n * m * p  # 63
    best_decomp = None
    
    # We prioritize finding the rank-35 solution (23 + 12)
    ranks_to_try = [35, 34, 33, 32, 31, 30]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try to find a rank-r decomposition using ALS with structured init
        decomp = _find_cp_decomposition_als(A_idx, B_idx, C_idx, nm, mp, np_, r, n, m, p, deadline - 5)
        if decomp is not None:
            best_rank = r
            best_decomp = decomp
            # If we find a very good one, we might stop, but let's try one lower
            if r <= 30:
                break
    
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _find_cp_decomposition_als(A_idx, B_idx, C_idx, nm, mp, np_, r, n, m, p, deadline):
    """
    Find a rank-r CP decomposition using Alternating Least Squares (ALS).
    """
    num_restarts = 50
    max_iters = 2000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            return None
        
        rng = np.random.default_rng(seed=1000 + restart)
        
        # Initialize U, V, W
        # Use structured initialization for rank 35
        U, V, W = _initialize_structured(nm, mp, np_, r, n, m, p, rng)
        
        prev_error = float('inf')
        converged = False
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            U = _als_update(U, V, W, A_idx, B_idx, C_idx, nm, r)
            # Update V
            V = _als_update(V, W, U, B_idx, C_idx, A_idx, mp, r)
            # Update W
            W = _als_update(W, U, V, C_idx, A_idx, B_idx, np_, r)
            
            # Compute error
            error = _compute_error(U, V, W, A_idx, B_idx, C_idx)
            
            if abs(prev_error - error) < 1e-10:
                break
            prev_error = error
        
        # Final check
        max_err = _max_error(U, V, W, A_idx, B_idx, C_idx)
        
        if max_err < 1e-8:
            # Check non-support entries (sample)
            sampled_err = 0.0
            for _ in range(200):
                a = rng.integers(0, nm)
                b = rng.integers(0, mp)
                c = rng.integers(0, np_)
                # Check if (a,b,c) is in support
                in_support = False
                for idx in range(len(A_idx)):
                    if A_idx[idx] == a and B_idx[idx] == b and C_idx[idx] == c:
                        in_support = True
                        break
                if not in_support:
                    approx = np.dot(U[a, :], V[b, :] * W[c, :])
                    sampled_err = max(sampled_err, abs(approx))
            
            if sampled_err < 1e-8:
                U_list = [[float(U[a, c]) for c in range(r)] for a in range(nm)]
                V_list = [[float(V[b, c]) for c in range(r)] for b in range(mp)]
                W_list = [[float(W[c, c]) for c in range(r)] for c in range(np_)]
                return (U_list, V_list, W_list)
    
    return None

def _initialize_structured(nm, mp, np_, r, n, m, p, rng):
    """
    Initialize U, V, W with a block structure.
    For r=35, we split into 23 (for k=0,1,2) and 12 (for k=3,4,5,6).
    """
    U = np.zeros((nm, r))
    V = np.zeros((mp, r))
    W = np.zeros((np_, r))
    
    if r == 35:
        r1 = 23
        r2 = 12
        # Block 1: k in [0, 1, 2]
        # W rows for k=0,1,2 are active
        for k in range(3):
            for i in range(n):
                idx_c = k * n + i
                W[idx_c, :r1] = rng.standard_normal(r1)
        
        # V rows for all j,k are active in block 1?
        # In the tensor T[i,j,k], V is indexed by j,k.
        # For the first block, we are solving for the sub-tensor where k is 0,1,2.
        # However, V[j,k] couples j and k.
        # To decouple, we can assume V[j,k] is non-zero for all j in block 1.
        for j in range(m):
            for k in range(p):
                idx_v = j * p + k
                V[idx_v, :r1] = rng.standard_normal(r1)
        
        # U rows for all i,j are active in block 1.
        for i in range(n):
            for j in range(m):
                idx_u = i * m + j
                U[idx_u, :r1] = rng.standard_normal(r1)
        
        # Block 2: k in [3, 4, 5, 6]
        # Schoolbook-like initialization
        col_offset = r1
        for i in range(n):
            for j in range(m):
                for k in range(3, p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_c = k * n + i
                    U[idx_u, col_offset] = 1.0
                    V[idx_v, col_offset] = 1.0
                    W[idx_c, col_offset] = 1.0
                    col_offset += 1
                    if col_offset >= r:
                        break
                if col_offset >= r:
                    break
            if col_offset >= r:
                break
        if col_offset >= r:
            break
    else:
        # Generic random initialization for other ranks
        U = rng.standard_normal((nm, r))
        V = rng.standard_normal((mp, r))
        W = rng.standard_normal((np_, r))
        
    return U, V, W

def _als_update(X, Y, Z, X_idx, Y_idx, Z_idx, dim_x, r):
    """
    Update X given Y, Z.
    X has shape (dim_x, r).
    X_idx, Y_idx, Z_idx are arrays of indices.
    """
    new_X = np.zeros_like(X)
    
    # Group indices by X_idx
    # For each x in range(dim_x):
    #   Find all entries where X_idx == x
    #   Solve for X[x, :]
    
    for x in range(dim_x):
        mask = (X_idx == x)
        if not np.any(mask):
            continue
        
        y_indices = Y_idx[mask]
        z_indices = Z_idx[mask]
        
        # Y_vals: shape (num_matches, r)
        Y_vals = Y[y_indices, :]
        Z_vals = Z[z_indices, :]
        
        # We want to minimize sum_t (sum_r X[x,r] Y[t,r] Z[t,r] - 1)^2
        # Let M = Y * Z (element-wise), shape (num_matches, r)
        M = Y_vals * Z_vals
        
        # Linear system: X[x, :] M^T = 1 (vector of ones)
        # X[x, :] (M^T M) = M^T 1
        # X[x, :] = (M^T 1) (M^T M)^{-1}
        
        # Add small regularization for stability
        reg = 1e-9 * np.eye(r)
        
        try:
            MTM = M.T @ M + reg
            MT1 = M.T @ np.ones(len(y_indices))
            new_X[x, :] = np.linalg.solve(MTM, MT1)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            new_X[x, :] = np.linalg.lstsq(M, np.ones(len(y_indices)), rcond=None)[0]
            
    return new_X

def _compute_error(U, V, W, A_idx, B_idx, C_idx):
    error = 0.0
    for idx in range(len(A_idx)):
        a = A_idx[idx]
        b = B_idx[idx]
        c = C_idx[idx]
        approx = np.dot(U[a, :], V[b, :] * W[c, :])
        error += (approx - 1.0) ** 2
    return np.sqrt(error / len(A_idx))

def _max_error(U, V, W, A_idx, B_idx, C_idx):
    max_err = 0.0
    for idx in range(len(A_idx)):
        a = A_idx[idx]
        b = B_idx[idx]
        c = C_idx[idx]
        approx = np.dot(U[a, :], V[b, :] * W[c, :])
        max_err = max(max_err, abs(approx - 1.0))
    return max_err
# EVOLVE-BLOCK-END
