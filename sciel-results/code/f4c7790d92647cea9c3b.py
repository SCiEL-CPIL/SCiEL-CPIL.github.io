"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _snap_to_half_integers(U, V, W):
    """Snap values to the nearest 0.5."""
    U_snapped = np.round(U * 2) / 2
    V_snapped = np.round(V * 2) / 2
    W_snapped = np.round(W * 2) / 2
    return U_snapped, V_snapped, W_snapped

def _als_decompose(T, rank, n, m, p, max_iters=500, tol=1e-12, init_U=None, init_V=None, init_W=None, deadline=None):
    """
    Alternating Least Squares for tensor decomposition.
    T: target tensor (n*m, m*p, p*n)
    rank: target rank
    Returns U, V, W if converged, else None.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    if init_U is None:
        U = np.random.randn(dim_a, rank)
        V = np.random.randn(dim_b, rank)
        W = np.random.randn(dim_c, rank)
    else:
        U = init_U.copy()
        V = init_V.copy()
        W = init_W.copy()
        
    prev_error = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None
            
        # Update U
        M_VW = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, rank)
        for a in range(dim_a):
            T_flat = T[a, :, :].reshape(-1)
            u_a, _, _, _ = np.linalg.lstsq(M_VW, T_flat, rcond=None)
            U[a, :] = u_a
            
        # Update V
        M_UW = np.einsum('ar,cr->acr', U, W).reshape(dim_a * dim_c, rank)
        for b in range(dim_b):
            T_flat = T[:, b, :].reshape(-1)
            v_b, _, _, _ = np.linalg.lstsq(M_UW, T_flat, rcond=None)
            V[b, :] = v_b
            
        # Update W
        M_UV = np.einsum('ar,br->abr', U, V).reshape(dim_a * dim_b, rank)
        for c in range(dim_c):
            T_flat = T[:, :, c].reshape(-1)
            w_c, _, _, _ = np.linalg.lstsq(M_UV, T_flat, rcond=None)
            W[c, :] = w_c
            
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        if abs(prev_error - error) < tol:
            if error < 1e-7:
                return U, V, W
            else:
                return None
        prev_error = error
        
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-7:
        return U, V, W
    return None

def _get_svd_init(T, rank, n, m, p):
    """Get SVD-based initialization for U, V, W."""
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    # Matricize T
    T_A = T.reshape(dim_a, dim_b * dim_c)
    T_B = T.reshape(dim_b, dim_a * dim_c).T # T[b, a, c] -> T_B[b, :]
    T_C = T.reshape(dim_c, dim_a * dim_b).T # T[c, a, b] -> T_C[c, :]
    
    # SVD
    U_svd, _, _ = np.linalg.svd(T_A, full_matrices=False)
    V_svd, _, _ = np.linalg.svd(T_B, full_matrices=False)
    W_svd, _, _ = np.linalg.svd(T_C, full_matrices=False)
    
    U_init = U_svd[:, :rank]
    V_init = V_svd[:, :rank]
    W_init = W_svd[:, :rank]
    
    return U_init, V_init, W_init

def _try_rank(target_rank, n, m, p, deadline, max_attempts=20, seed=42):
    """Try to find a decomposition of given rank using ALS with SVD init."""
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    U0, V0, W0 = _get_svd_init(T, target_rank, n, m, p)
    
    rng = random.Random(seed)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Perturb SVD init slightly to escape potential symmetries
        perturb_scale = 0.1
        init_U = U0 + rng.uniform(-perturb_scale, perturb_scale, U0.shape)
        init_V = V0 + rng.uniform(-perturb_scale, perturb_scale, V0.shape)
        init_W = W0 + rng.uniform(-perturb_scale, perturb_scale, W0.shape)
        
        result = _als_decompose(T, target_rank, n, m, p, max_iters=200, tol=1e-10,
                                init_U=init_U, init_V=init_V, init_W=init_W,
                                deadline=deadline)
        
        if result is not None:
            U, V, W = result
            # Snap to half-integers
            U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
            
            # Verify snapped version
            if _verify_fast(U_s, V_s, W_s, n, m, p, tol=1e-9):
                return U_s, V_s, W_s
            
            # Verify un-snapped version (just in case)
            if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                return U, V, W
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Try rank 23 first
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        result = _try_rank(target_rank, n, m, p, deadline, max_attempts=30, seed=42 + target_rank)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
