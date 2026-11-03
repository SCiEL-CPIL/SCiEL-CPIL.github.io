import time
import numpy as np
import scipy.optimize

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

def _verify_exact(U, V, W, n, m, p, T_target, tol=1e-10):
    """
    Verify if U, V, W form a valid decomposition of T_target.
    Also checks if entries are close to integers or half-integers.
    """
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
        
    # Check if entries are integer or half-integer
    # The problem states "integer/half-integer entries".
    # We check if 2*U, 2*V, 2*W are close to integers.
    if not (np.allclose(U * 2, np.round(U * 2), atol=1e-6) and
            np.allclose(V * 2, np.round(V * 2), atol=1e-6) and
            np.allclose(W * 2, np.round(W * 2), atol=1e-6)):
        return False

    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    # Exact check (using float is risky for 'exact', but we check residual)
    T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
    
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
        
    return True, U_snap, V_snap, W_snap

def _try_rank_23(n, m, p, T_target, max_seeds=100, seed_base=1000):
    """
    Attempt to find a rank 23 decomposition.
    Uses Levenberg-Marquardt.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    rank = 23
    
    # Total variables: (6 + 12 + 8) * 23 = 26 * 23 = 598
    # Total residuals: 6 * 12 * 8 = 576
    # Overdetermined, so LM is appropriate.
    
    def residual(x):
        U = x[:dim_U * rank].reshape(dim_U, rank)
        V = x[dim_U * rank : dim_U * rank + dim_V * rank].reshape(dim_V, rank)
        W = x[dim_U * rank + dim_V * rank :].reshape(dim_W, rank)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_recon - T_target).flatten()
    
    for seed in range(max_seeds):
        if time.time() > _DEADLINE:
            return None
            
        # Structured initialization: Start with schoolbook and add noise to some columns
        # Schoolbook has 24 columns. We need 23.
        # We can try to "merge" two schoolbook terms or just random init.
        # Random init is often better for escaping local minima in these high-dim problems.
        
        rng = np.random.RandomState(seed_base + seed)
        
        # Scale of init matters. 
        # The entries of T are 1. 
        # If U, V, W are ~ N(0, 1), then T_recon entries are sums of products.
        # We want T_recon ~ 1.
        
        U_init = rng.randn(dim_U, rank) * 0.5
        V_init = rng.randn(dim_V, rank) * 0.5
        W_init = rng.randn(dim_W, rank) * 0.5
        
        # Add a small component of the schoolbook pattern to guide it?
        # Let's try mixing.
        if seed % 2 == 0:
             # Pure random
             pass
        else:
             # Structured: take first 23 schoolbook columns and perturb
             U_init = np.zeros((dim_U, rank))
             V_init = np.zeros((dim_V, rank))
             W_init = np.zeros((dim_W, rank))
             
             # Fill with schoolbook pattern for 23 terms
             count = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         if count >= rank: break
                         U_init[i*m+j, count] = 1.0
                         V_init[j*p+k, count] = 1.0
                         W_init[k*n+i, count] = 1.0
                         count += 1
                         if count >= rank: break
                     if count >= rank: break
                 if count >= rank: break
             
             # Perturb
             U_init += rng.randn(dim_U, rank) * 0.1
             V_init += rng.randn(dim_V, rank) * 0.1
             W_init += rng.randn(dim_W, rank) * 0.1
             
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = scipy.optimize.least_squares(
                residual, x0,
                method='lm',
                ftol=1e-12, xtol=1e-12, gtol=1e-12,
                max_nfev=2000
            )
        except Exception:
            continue
            
        if res.cost < 1e-10:
            x_sol = res.x
            U = x_sol[:dim_U * rank].reshape(dim_U, rank)
            V = x_sol[dim_U * rank : dim_U * rank + dim_V * rank].reshape(dim_V, rank)
            W = x_sol[dim_U * rank + dim_V * rank :].reshape(dim_W, rank)
            
            # Verify and snap
            result = _verify_exact(U, V, W, n, m, p, T_target)
            if result[0]:
                return result[1], result[2], result[3]
                
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
    
    # Try to find rank 23
    # This is the only rank < 24 that is feasible for LM with this variable count.
    # Ranks < 22 are underdetermined for LM and unlikely.
    res = _try_rank_23(n, m, p, T_target, max_seeds=50, seed_base=42)
    
    if res is not None:
        U, V, W = res
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: Schoolbook (Rank 24)
    U, V, W = _get_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
