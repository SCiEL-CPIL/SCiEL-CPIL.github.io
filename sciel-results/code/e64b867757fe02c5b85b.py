import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]
RNG_SEED = 42

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _build_rank31_init(U32, V32, W32, seed, mode=0):
    """Create a rank-31 initialization using various strategies."""
    rng = np.random.default_rng(seed)
    rank = 32
    
    if mode == 0:
        # Drop a column from a trivial block (columns 28-31)
        j = rng.integers(28, 32)
        U31 = np.delete(U32, j, axis=1)
        V31 = np.delete(V32, j, axis=1)
        W31 = np.delete(W32, j, axis=1)
        noise_scale = 0.05
        U31 = U31 + rng.normal(0, noise_scale, size=U31.shape)
        V31 = V31 + rng.normal(0, noise_scale, size=V31.shape)
        W31 = W31 + rng.normal(0, noise_scale, size=W31.shape)
    elif mode == 1:
        # Drop a column from a Strassen block
        j = rng.integers(0, 28)
        U31 = np.delete(U32, j, axis=1)
        V31 = np.delete(V32, j, axis=1)
        W31 = np.delete(W32, j, axis=1)
        noise_scale = 0.05
        U31 = U31 + rng.normal(0, noise_scale, size=U31.shape)
        V31 = V31 + rng.normal(0, noise_scale, size=V31.shape)
        W31 = W31 + rng.normal(0, noise_scale, size=W31.shape)
    elif mode == 2:
        # Random linear combination: create 31 columns from 32
        # Pick a random 32x31 matrix and multiply
        M = rng.normal(0, 1, size=(rank, rank-1))
        M /= np.linalg.norm(M, axis=0, keepdims=True)
        U31 = U32 @ M
        V31 = V32 @ M
        W31 = W32 @ M
    elif mode == 3:
        # Structured: merge two columns by averaging, drop one
        i, j = rng.choice(rank, size=2, replace=False)
        U31 = U32.copy()
        V31 = V32.copy()
        W31 = W32.copy()
        # Average columns i and j into column i
        U31[:, i] = (U32[:, i] + U32[:, j]) / 2
        V31[:, i] = (V32[:, i] + V32[:, j]) / 2
        W31[:, i] = (W32[:, i] + W32[:, j]) / 2
        # Remove column j
        U31 = np.delete(U31, j, axis=1)
        V31 = np.delete(V31, j, axis=1)
        W31 = np.delete(W31, j, axis=1)
        noise_scale = 0.03
        U31 = U31 + rng.normal(0, noise_scale, size=U31.shape)
        V31 = V31 + rng.normal(0, noise_scale, size=V31.shape)
        W31 = W31 + rng.normal(0, noise_scale, size=W31.shape)
    else:
        # Pure random
        U31 = rng.normal(0, 1, size=(U32.shape[0], 31))
        V31 = rng.normal(0, 1, size=(V32.shape[0], 31))
        W31 = rng.normal(0, 1, size=(W32.shape[0], 31))
    
    return U31, V31, W31

def _compute_residual(U, V, W, n, m, p):
    """Compute the maximum absolute residual of the tensor decomposition."""
    rank = U.shape[1]
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T_true = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    residual = M - T_true
    return np.max(np.abs(residual))

def _soft_max_abs(residual, beta=100.0):
    """Compute a smooth approximation of max(|residual|) using log-sum-exp."""
    abs_res = np.abs(residual)
    # Soft max via log-sum-exp: (1/beta) * log(sum(exp(beta * x)))
    # For numerical stability, subtract max
    max_val = np.max(abs_res)
    if max_val < 1e-15:
        return 0.0
    shifted = abs_res - max_val
    log_sum_exp = np.log(np.sum(np.exp(beta * shifted))) + beta * max_val
    return log_sum_exp / beta

def _try_rank31_search(U0, V0, W0, n, m, p, time_budget):
    """Try to find a rank-31 decomposition using direct optimization with multiple restarts."""
    start_time = time.time()
    
    nm, mp, pn = n * m, m * p, p * n
    rank31 = 31
    
    # Precompute true tensor
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    best_score = float('inf')
    best_result = None
    
    # Use multiple initialization modes
    modes = [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    
    for restart in range(len(modes)):
        if time.time() - start_time > time_budget * 0.85:
            break
        
        mode = modes[restart]
        # Create rank-31 initialization
        U31, V31, W31 = _build_rank31_init(U0, V0, W0, RNG_SEED + restart * 1000, mode)
        
        # Direct optimization of rank-31 problem
        x0 = np.concatenate([U31.flatten(), V31.flatten(), W31.flatten()])
        
        def objective(params):
            U_flat = params[:nm * rank31]
            V_flat = params[nm * rank31 : nm * rank31 + mp * rank31]
            W_flat = params[nm * rank31 + mp * rank31 : nm * rank31 + mp * rank31 + pn * rank31]
            
            U = U_flat.reshape(nm, rank31)
            V = V_flat.reshape(mp, rank31)
            W = W_flat.reshape(pn, rank31)
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = M - T_true
            
            return np.sum(residual**2)
        
        try:
            # First pass: moderate tolerance
            result = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            
            if result.fun < 1e-4:
                # Second pass: tighter tolerance
                result2 = minimize(objective, result.x, method='L-BFGS-B',
                                   options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80})
                
                if result2.fun < 1e-8:
                    U_flat = result2.x[:nm * rank31]
                    V_flat = result2.x[nm * rank31 : nm * rank31 + mp * rank31]
                    W_flat = result2.x[nm * rank31 + mp * rank31 : nm * rank31 + mp * rank31 + pn * rank31]
                    U_sol = U_flat.reshape(nm, rank31)
                    V_sol = V_flat.reshape(mp, rank31)
                    W_sol = W_flat.reshape(pn, rank31)
                    
                    score = _compute_residual(U_sol, V_sol, W_sol, n, m, p)
                    if score < 1e-6 and score < best_score:
                        best_score = score
                        best_result = (U_sol.copy(), V_sol.copy(), W_sol.copy(), score)
        except Exception:
            continue
    
    return best_result

def _polish_rank31(U, V, W, n, m, p, T_true, time_budget):
    """High-precision polish pass for rank-31 decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    start_time = time.time()
    
    def objective(params):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = M - T_true
        
        # Use sum of squares for smooth optimization
        return np.sum(residual**2)
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
    except Exception:
        return U, V, W
    
    if result.fun < 1e-10:
        U_flat = result.x[:nm * rank]
        V_flat = result.x[nm * rank : nm * rank + mp * rank]
        W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        U_polished = U_flat.reshape(nm, rank)
        V_polished = V_flat.reshape(mp, rank)
        W_polished = W_flat.reshape(pn, rank)
        return U_polished, V_polished, W_polished
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First, build the rank-32 solution
    U32, V32, W32 = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    # Try to find rank-31 using direct optimization with multiple restarts
    time_budget = min(60.0, deadline - time.time())
    best_result = _try_rank31_search(U32, V32, W32, n, m, p, time_budget)
    
    if best_result is not None:
        U_sol, V_sol, W_sol, score = best_result
        return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fall back to rank-32
    return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}
