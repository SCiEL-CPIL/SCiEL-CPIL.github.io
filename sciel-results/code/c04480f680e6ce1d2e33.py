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

def _build_target_tensor(n, m, p):
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _als_decompose(T, rank, n, m, p, max_iters=500, tol=1e-10, init_U=None, init_V=None, init_W=None, deadline=None):
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
        
    # Normalize initial vectors for stability
    for r in range(rank):
        nu = np.linalg.norm(U[:, r])
        nv = np.linalg.norm(V[:, r])
        nw = np.linalg.norm(W[:, r])
        if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
            U[:, r] /= nu
            V[:, r] /= nv
            W[:, r] /= nw
            
    prev_error = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None
            
        # Update U: M[bc, r] = V[b,r] * W[c,r], computed ONCE
        M = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, rank)
        for a in range(dim_a):
            T_flat = T[a, :, :].reshape(-1)
            u_a, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            U[a, :] = u_a
            
        # Update V: M[ac, r] = U[a,r] * W[c,r], computed ONCE
        M = np.einsum('ar,cr->acr', U, W).reshape(dim_a * dim_c, rank)
        for b in range(dim_b):
            T_flat = T[:, b, :].reshape(-1)
            v_b, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            V[b, :] = v_b
            
        # Update W: M[ab, r] = U[a,r] * V[b,r], computed ONCE
        M = np.einsum('ar,br->abr', U, V).reshape(dim_a * dim_b, rank)
        for c in range(dim_c):
            T_flat = T[:, :, c].reshape(-1)
            w_c, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            W[c, :] = w_c
            
        # Calculate error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        if abs(prev_error - error) < tol:
            if error < 1e-7:
                return U, V, W
            else:
                return None
        prev_error = error
        
    # Final check
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-7:
        return U, V, W
    return None

def _try_rank(target_rank, n, m, p, deadline, max_attempts=100, seed=42):
    """
    Attempt to find a decomposition of given rank using ALS.
    Tries dropping (27 - target_rank) terms from schoolbook with perturbations.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = _build_target_tensor(n, m, p)
    
    U0, V0, W0 = _schoolbook(n, m, p)
    total_terms = n*m*p
    n_drop = total_terms - target_rank
    
    if n_drop <= 0:
        return U0, V0, W0
    
    all_indices = list(range(total_terms))
    rng = random.Random(seed)
    
    # Try diverse initialization strategies
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Pick indices to drop
        drop_indices = rng.sample(all_indices, n_drop)
        keep_indices = [i for i in all_indices if i not in drop_indices]
        
        init_U = U0[:, keep_indices].copy()
        init_V = V0[:, keep_indices].copy()
        init_W = W0[:, keep_indices].copy()
        
        # Add small random perturbation to help escape local minima
        perturb_scale = rng.uniform(0.05, 0.3)
        for r_idx in range(len(keep_indices)):
            mask_u = init_U[:, r_idx] != 0
            mask_v = init_V[:, r_idx] != 0
            mask_w = init_W[:, r_idx] != 0
            if mask_u.any():
                init_U[mask_u, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            if mask_v.any():
                init_V[mask_v, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            if mask_w.any():
                init_W[mask_w, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            
        # Run ALS
        result = _als_decompose(T, target_rank, n, m, p, max_iters=300, tol=1e-10,
                                init_U=init_U, init_V=init_V, init_W=init_W,
                                deadline=deadline)
        
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                return U, V, W
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Try ranks in order of increasing difficulty
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        result = _try_rank(target_rank, n, m, p, deadline, max_attempts=60, seed=42)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
