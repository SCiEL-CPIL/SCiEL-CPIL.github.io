import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U,V,W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Construct target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    # Compute reconstructed tensor
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check difference
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _try_rank(rank, n, m, p, max_time=10, seed=42):
    """Try to find a rank-rank decomposition via optimization."""
    rng = np.random.RandomState(seed)
    
    # Dimensions
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Initialize
    U = rng.randn(dim_U, rank)
    V = rng.randn(dim_V, rank)
    W = rng.randn(dim_W, rank)
    
    # Target tensor
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Flatten parameters
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def loss(x):
        if time.time() > _DEADLINE:
            return 1e10
        U_flat, V_flat, W_flat = np.split(x, [dim_U*rank, dim_U*rank + dim_V*rank])
        U = U_flat.reshape(dim_U, rank)
        V = V_flat.reshape(dim_V, rank)
        W = W_flat.reshape(dim_W, rank)
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_recon - T_target
        return np.sum(diff**2)
        
    def jac(x):
        # Numerical gradient might be too slow. Use finite differences if needed, or let scipy estimate.
        # For this size, scipy's finite difference is okay for a few iterations.
        # But let's provide an analytical gradient if possible?
        # Gradient of sum (sum_r U_ar V_br W_cr - T_abc)^2
        # dL/dU_ar = 2 * sum_bc (sum_r' U_ar' V_br' W_cr' - T_abc) V_br W_cr
        # This is computable.
        U_flat, V_flat, W_flat = np.split(x, [dim_U*rank, dim_U*rank + dim_V*rank])
        U = U_flat.reshape(dim_U, rank)
        V = V_flat.reshape(dim_V, rank)
        W = W_flat.reshape(dim_W, rank)
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_recon - T_target # (A, B, C)
        
        # dL/dU_ar = 2 * sum_bc diff_abc V_br W_cr
        # V_br W_cr -> (B, C) for each r? No, sum over r.
        # Let M_U_ar = sum_bc diff_abc V_br W_cr
        # This is a contraction.
        # diff (A,B,C), V (B,R), W (C,R)
        # We want (A,R).
        # M_U = np.einsum('abc,br,cr->ar', diff, V, W)
        
        # Similarly for V:
        # M_V = np.einsum('abc,ar,cr->br', diff, U, W)
        
        # For W:
        # M_W = np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # Optimize
    result = None
    try:
        result = scipy.optimize.minimize(loss, x0, jac=jac, method='L-BFGS-B', 
                                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8})
    except Exception:
        pass
        
    if result is not None and result.fun < 1e-8:
        # Try to round to half-integers
        U_opt, V_opt, W_opt = np.split(result.x, [dim_U*rank, dim_U*rank + dim_V*rank])
        U_opt = U_opt.reshape(dim_U, rank)
        V_opt = V_opt.reshape(dim_V, rank)
        W_opt = W_opt.reshape(dim_W, rank)
        
        U_rounded = np.round(U_opt * 2) / 2
        V_rounded = np.round(V_opt * 2) / 2
        W_rounded = np.round(W_opt * 2) / 2
        
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
            
        # If rounding fails, maybe the solution is exact integers?
        U_int = np.round(U_opt)
        V_int = np.round(V_opt)
        W_int = np.round(W_opt)
        if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
            return (U_int.tolist(), V_int.tolist(), W_int.tolist())
            
    return None

def construct():
    n, m, p = 2, 3, 4
    lower_bound = max(n*m, m*p, n*p)
    
    # Try ranks from lower_bound up to 24
    for rank in range(lower_bound, 25):
        if time.time() > _DEADLINE - 5:
            break
            
        # Try multiple seeds
        for seed in range(20):
            if time.time() > _DEADLINE - 2:
                break
            res = _try_rank(rank, n, m, p, seed=seed)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
                
    # Fallback: Schoolbook
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
