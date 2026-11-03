"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _build_rank88():
    """Build the known rank-88 decomposition."""
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    strassen_terms = [
        {'U': {0: 1, 3: 1}, 'V': {0: 1, 3: 1}, 'W': {0: 1, 3: 1}},
        {'U': {2: 1, 3: 1}, 'V': {0: 1}, 'W': {2: 1, 3: -1}},
        {'U': {0: 1}, 'V': {1: 1, 3: -1}, 'W': {1: 1, 3: 1}},
        {'U': {3: 1}, 'V': {2: 1, 0: -1}, 'W': {0: 1, 2: 1}},
        {'U': {0: 1, 1: 1}, 'V': {3: 1}, 'W': {0: -1, 1: 1}},
        {'U': {2: 1, 0: -1}, 'V': {0: 1, 1: 1}, 'W': {3: 1}},
        {'U': {1: 1, 3: -1}, 'V': {2: 1, 3: 1}, 'W': {0: 1}}
    ]
    
    for j_block in range(4):
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                U[(1 + lu) * m + lv][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                V[lu * p + (2 * j_block + lv)][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                W[(2 * j_block + lv) * n + (1 + lu)][col] += val
            col += 1
            
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                U[(1 + lu) * m + (2 + lv)][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                V[(2 + lu) * p + (2 * j_block + lv)][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                W[(2 * j_block + lv) * n + (1 + lu)][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Reconstruct C using the decomposition: C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    C_reconstructed = np.zeros((n, p))
    for r in range(U.shape[1]):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        # C[i,k] += sum_j u_r[i,j] * v_r[j,k] * w_r[k,i]
        for i in range(n):
            for k in range(p):
                C_reconstructed[i, k] += w_r[k, i] * np.dot(u_r[i, :], v_r[:, k])
    
    return np.allclose(C_expected, C_reconstructed, atol=1e-9)

def _als_decomposition(target_rank, n_iter=200, seed=0, init=None):
    """ALS for tensor decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    U_size = n * m  # 12
    V_size = m * p  # 32
    W_size = p * n  # 24
    
    # Precompute the tensor T as a 3D array T[i,j,k] = 1 for all valid (i,j,k)
    # Actually for matrix mult: T[i,j,k] = 1 always (since j is the shared index)
    # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Precompute index arrays
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    idx_u = np.array([i * m + j for i, j, k in indices])
    idx_v = np.array([j * p + k for i, j, k in indices])
    idx_w = np.array([k * n + i for i, j, k in indices])
    
    # Initialize
    if init is not None:
        U, V, W = init
    else:
        U = rng.randn(U_size, target_rank) * 0.5
        V = rng.randn(V_size, target_rank) * 0.5
        W = rng.randn(W_size, target_rank) * 0.5
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(n_iter):
        if time.time() > _START_TIME + _DEADLINE - _MARGIN:
            break
        
        # Update U: for each column r, solve least squares
        # Fix V, W. For each r: U[:,r] minimizes sum_{i,j,k} (T[i,j,k] - sum_{r'} U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r'])^2
        # This is a coupled problem. Use ALS: fix V,W, solve for U via linear system.
        
        # For fixed V and W, the contribution of U[:,r] to the residual is:
        # residual = T - sum_r U[idx_u, r] * (V[idx_v, r] * W[idx_w, r])
        # So we need: sum_r U[idx_u, r] * B_r = T where B_r[i,j,k] = V[idx_v[r], r] * W[idx_w[r], r]
        
        # Build the linear system: A_mat @ U_flat = T_flat
        # A_mat has shape (n*m*p, U_size * target_rank)
        # This is too large. Instead, do per-column updates.
        
        # Alternative: for each r, compute the gradient and do a step.
        # Or: fix V and W, then U[:,r] appears in the sum only through its coupling with V[:,r] and W[:,r].
        
        # Actually, proper ALS: fix V and W, then for each r independently:
        # The term involving U[:,r] is: U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
        # So define D_r = V[idx_v, r] * W[idx_w, r] (shape n*m*p)
        # Then we want: sum_r U[idx_u, r] * D_r = T
        # This is: M @ U_vec = T_vec where M is (n*m*p) x (U_size * rank)
        
        # M[i*m*p + (i,j,k), (i'*m+j')*rank + r] = D_r[i,j,k] if i*m+j == i'*m+j' else 0
        # This is block diagonal! Each block corresponds to a fixed (i,j) pair.
        
        # For fixed (i,j): the equations are over k: sum_r U[i*m+j, r] * D_r[i,j,k] = T[i,j,k]
        # So for each (i,j), we have a p x rank system.
        
        # Let's do this properly:
        # For each (i,j), define d_ij = V[idx_v for this (i,j)], W[idx_w for this (i,j)]
        # Actually, idx_v depends on (j,k) and idx_w depends on (k,i)
        
        # For fixed (i,j): 
        #   U_row = U[i*m+j, :]  (length rank)
        #   For each k: sum_r U_row[r] * V[j*p+k, r] * W[k*n+i, r] = 1
        #   This is: (V_j_row * W_i_row) @ U_row = ones(p)
        #   where V_j_row = V[j*p:(j+1)*p, :] and W_i_row = W[:, i] reshaped... 
        #   Actually W[k*n+i, r] for varying k gives W[0*n+i, r], W[1*n+i, r], ..., W[(p-1)*n+i, r]
        
        # Let's precompute for efficiency:
        # For each (i,j): 
        #   V_block = V[j*p:(j+1)*p, :]  (p x rank)
        #   W_block = W[::n, i] ... no, W[k*n+i] for k=0..p-1
        #   W_vec = np.array([W[k*n+i, :] for k in range(p)])  (p x rank)
        #   Element-wise product: P = V_block * W_vec  (p x rank)
        #   Solve: P @ U[i*m+j, :] = ones(p)
        
        # This is a least squares problem: min ||P @ u - 1||^2
        # u = P^T (P P^T)^{-1} 1 if P P^T is invertible
        
        # Update U:
        for i in range(n):
            for j in range(m):
                V_block = V[j * p:(j + 1) * p, :]  # (p, rank)
                W_vec = np.array([W[k * n + i, :] for k in range(p)])  # (p, rank)
                P = V_block * W_vec  # (p, rank)
                # Solve P @ u = ones(p)
                try:
                    # Use lstsq
                    u, _, _, _ = np.linalg.lstsq(P, np.ones(p), rcond=None)
                    U[i * m + j, :] = u
                except:
                    pass
        
        # Update V similarly:
        for j in range(m):
            for k in range(p):
                U_block = np.array([U[i * m + j, :] for i in range(n)])  # (n, rank)
                W_vec = W[k * n:(k + 1) * n, :]  # (n, rank)
                P = U_block * W_vec  # (n, rank)
                try:
                    v, _, _, _ = np.linalg.lstsq(P, np.ones(n), rcond=None)
                    V[j * p + k, :] = v
                except:
                    pass
        
        # Update W similarly:
        for k in range(p):
            for i in range(n):
                U_block = np.array([U[i * m + j, :] for j in range(m)])  # (m, rank)
                V_vec = np.array([V[j * p + k, :] for j in range(m)])  # (m, rank)
                P = U_block * V_vec  # (m, rank)
                try:
                    w, _, _, _ = np.linalg.lstsq(P, np.ones(m), rcond=None)
                    W[k * n + i, :] = w
                except:
                    pass
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-10:
            break
    
    # Check if solution is exact
    if best_error < 1e-8:
        # Round to half-integers
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        
        # Verify rounded solution
        error_r = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                    error_r += (val - 1.0) ** 2
        
        if error_r < 1e-8:
            return U_r.tolist(), V_r.tolist(), W_r.tolist(), error_r
        
        # Try integer rounding too
        U_i = np.round(best_U).astype(float)
        V_i = np.round(best_V).astype(float)
        W_i = np.round(best_W).astype(float)
        error_i = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_i[i*m+j, :], V_i[j*p+k, :] * W_i[k*n+i, :])
                    error_i += (val - 1.0) ** 2
        if error_i < 1e-8:
            return U_i.tolist(), V_i.tolist(), W_i.tolist(), error_i
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, get the guaranteed rank-88 solution
    rank88_result = _build_rank88()
    
    # Try to find a better solution
    for target_rank in [86, 84, 82, 80]:
        time_remaining = _START_TIME + _DEADLINE - _MARGIN - time.time()
        if time_remaining < 15:
            break
        
        max_iters = min(150, int(time_remaining * 2))
        for seed in range(3):
            time_remaining = _START_TIME + _DEADLINE - _MARGIN - time.time()
            if time_remaining < 10:
                break
            
            result = _als_decomposition(target_rank, n_iter=max_iters, seed=seed)
            if result is not None:
                U, V, W, err = result
                if _verify_decomposition(U, V, W):
                    return {"U": U, "V": V, "W": W}
    
    # Return the guaranteed rank-88 solution
    return rank88_result
# EVOLVE-BLOCK-END
