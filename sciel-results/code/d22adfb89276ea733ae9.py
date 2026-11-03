import time
import numpy as np
import scipy.optimize
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin for 150s limit

def _build_target_tensor(n, m, p):
    """Construct the target matrix multiplication tensor."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_fast(U, V, W, n, m, p, T_target, tol=1e-10):
    """
    Fast verification: check if U, V, W form a valid decomposition.
    Returns (valid, U_snap, V_snap, W_snap) if valid, else (False, None, None, None).
    """
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False, None, None, None
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False, None, None, None
        
    # Check if entries are close to integers or half-integers
    if not (np.allclose(U * 2, np.round(U * 2), atol=1e-6) and
            np.allclose(V * 2, np.round(V * 2), atol=1e-6) and
            np.allclose(W * 2, np.round(W * 2), atol=1e-6)):
        return False, None, None, None

    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    # Fast residual check
    T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
    residual = np.max(np.abs(T_recon - T_target))
    
    if residual > tol:
        return False, None, None, None
        
    return True, U_snap, V_snap, W_snap

def _polish(U, V, W, T_target, max_iter=100, tol=1e-15):
    """
    High-precision polish pass using ALS with tight tolerances.
    """
    dim_U, rank = U.shape
    dim_V = V.shape[0]
    dim_W = W.shape[0]
    
    for _ in range(max_iter):
        # Update U
        for a in range(dim_U):
            # U[a, :] = argmin sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # Linear system: V^T W^T U[a,:]^T = T[a,:,:].reshape(-1)
            VW = np.einsum('br,cr->bc', V, W)
            rhs = T_target[a, :, :].reshape(-1)
            try:
                U[a, :] = np.linalg.lstsq(VW, rhs, rcond=None)[0]
            except:
                pass
        
        # Update V
        for b in range(dim_V):
            UW = np.einsum('ar,cr->ac', U, W)
            rhs = T_target[:, b, :].reshape(-1)
            try:
                V[b, :] = np.linalg.lstsq(UW, rhs, rcond=None)[0]
            except:
                pass
        
        # Update W
        for c in range(dim_W):
            UV = np.einsum('ar,br->ab', U, V)
            rhs = T_target[:, :, c].reshape(-1)
            try:
                W[c, :] = np.linalg.lstsq(UV, rhs, rcond=None)[0]
            except:
                pass
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(T_recon - T_target)) < tol:
            break
    
    return U, V, W

def _try_als(n, m, p, T_target, rank, max_seeds=20, seed_base=42, max_iter=50):
    """
    Attempt to find a decomposition using ALS with multiple seeds.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    for seed in range(max_seeds):
        if time.time() > _DEADLINE:
            return None
            
        rng = np.random.RandomState(seed_base + seed)
        
        # Random initialization
        U = rng.randn(dim_U, rank) * 0.5
        V = rng.randn(dim_V, rank) * 0.5
        W = rng.randn(dim_W, rank) * 0.5
        
        # ALS iterations
        for _ in range(max_iter):
            if time.time() > _DEADLINE:
                return None
            
            # Update U
            for a in range(dim_U):
                VW = np.einsum('br,cr->bc', V, W)
                rhs = T_target[a, :, :].reshape(-1)
                try:
                    U[a, :] = np.linalg.lstsq(VW, rhs, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for b in range(dim_V):
                UW = np.einsum('ar,cr->ac', U, W)
                rhs = T_target[:, b, :].reshape(-1)
                try:
                    V[b, :] = np.linalg.lstsq(UW, rhs, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for c in range(dim_W):
                UV = np.einsum('ar,br->ab', U, V)
                rhs = T_target[:, :, c].reshape(-1)
                try:
                    W[c, :] = np.linalg.lstsq(UV, rhs, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.max(np.abs(T_recon - T_target))
            if residual < 1e-10:
                break
        
        # Verify and snap
        valid, U_snap, V_snap, W_snap = _verify_fast(U, V, W, n, m, p, T_target)
        if valid:
            return U_snap, V_snap, W_snap
            
    return None

def _try_rank(n, m, p, T_target, rank, max_seeds=10, seed_base=1000):
    """
    Attempt to find a decomposition of the given rank.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    def residual(x):
        U = x[:dim_U * rank].reshape(dim_U, rank)
        V = x[dim_U * rank : dim_U * rank + dim_V * rank].reshape(dim_V, rank)
        W = x[dim_U * rank + dim_V * rank :].reshape(dim_W, rank)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_recon - T_target).flatten()
    
    for seed in range(max_seeds):
        if time.time() > _DEADLINE:
            return None
            
        rng = np.random.RandomState(seed_base + seed)
        
        # Random initialization
        U_init = rng.randn(dim_U, rank) * 0.5
        V_init = rng.randn(dim_V, rank) * 0.5
        W_init = rng.randn(dim_W, rank) * 0.5
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = scipy.optimize.least_squares(
                residual, x0,
                method='lm',
                ftol=1e-12, xtol=1e-12, gtol=1e-12,
                max_nfev=1000
            )
        except Exception:
            continue
            
        if res.cost < 1e-10:
            x_sol = res.x
            U = x_sol[:dim_U * rank].reshape(dim_U, rank)
            V = x_sol[dim_U * rank : dim_U * rank + dim_V * rank].reshape(dim_V, rank)
            W = x_sol[dim_U * rank + dim_V * rank :].reshape(dim_W, rank)
            
            # Polish with ALS
            U, V, W = _polish(U, V, W, T_target, max_iter=50)
            
            # Verify and snap
            valid, U_snap, V_snap, W_snap = _verify_fast(U, V, W, n, m, p, T_target)
            if valid:
                return U_snap, V_snap, W_snap
                
    return None

def _get_schoolbook(n, m, p):
    """Return the standard schoolbook decomposition."""
    rank = n * m * p
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def construct():
    n, m, p = 2, 3, 4
    
    T_target = _build_target_tensor(n, m, p)
    
    # Try rank 13 with ALS
    res_13 = _try_als(n, m, p, T_target, 13, max_seeds=10, seed_base=100, max_iter=30)
    if res_13 is not None:
        U, V, W = res_13
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 23 with ALS
    res_23 = _try_als(n, m, p, T_target, 23, max_seeds=30, seed_base=42, max_iter=30)
    if res_23 is not None:
        U, V, W = res_23
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 23 with least_squares + polish
    res_23_ls = _try_rank(n, m, p, T_target, 23, max_seeds=20, seed_base=42)
    if res_23_ls is not None:
        U, V, W = res_23_ls
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: Schoolbook (Rank 24)
    U, V, W = _get_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
