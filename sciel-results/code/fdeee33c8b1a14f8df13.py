import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 9]

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

def _try_rank_optimization(target_rank, n, m, p, T_true, time_budget, rng_seed=42):
    """
    Attempt to find a decomposition of the given target_rank by optimizing
    from multiple random initializations and structured initializations.
    Uses least_squares for better performance.
    """
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    rank = target_rank
    
    def _residuals(params):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        return (M - T_true).flatten()

    best_result = None
    best_score = float('inf')
    
    # Generate initializations
    U32, V32, W32 = _build_rank32()
    
    inits = []
    
    # Structured init: Take rank 32, drop random columns, scale up
    for _ in range(10):
        idx = np.random.choice(32, size=32-target_rank, replace=False)
        keep = [i for i in range(32) if i not in idx]
        U_init = U32[:, keep]
        V_init = V32[:, keep]
        W_init = W32[:, keep]
        # Add small noise to break symmetry
        U_init = U_init + 0.1 * np.random.randn(nm, rank)
        V_init = V_init + 0.1 * np.random.randn(mp, rank)
        W_init = W_init + 0.1 * np.random.randn(pn, rank)
        inits.append(np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()]))
        
    # Pure random inits
    for _ in range(20):
        U_init = np.random.randn(nm, rank)
        V_init = np.random.randn(mp, rank)
        W_init = np.random.randn(pn, rank)
        inits.append(np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()]))
        
    for x0 in inits:
        if time.time() - start_time > time_budget * 0.9:
            break
            
        try:
            res = least_squares(_residuals, x0, method='trf', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            score = res.cost
            if score < best_score:
                best_score = score
                best_result = res.x
                if best_score < 1e-8:
                    break
                    
        except Exception:
            continue
            
    if best_result is not None and best_score < 1e-6:
        U_flat = best_result[:nm * rank]
        V_flat = best_result[nm * rank : nm * rank + mp * rank]
        W_flat = best_result[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        return U, V, W
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute T_true
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0

    def to_list(arr):
        return arr.tolist()
        
    # Try rank 30 first (ambitious)
    remaining = deadline - time.time()
    if remaining > 20:
        result_30 = _try_rank_optimization(30, n, m, p, T_true, min(remaining - 5, 40), rng_seed=1)
        if result_30 is not None:
            U30, V30, W30 = result_30
            score = _compute_residual(U30, V30, W30, n, m, p)
            if score < 1e-6:
                return {"U": to_list(U30), "V": to_list(V30), "W": to_list(W30)}
                
    # Try rank 31
    remaining = deadline - time.time()
    if remaining > 20:
        result_31 = _try_rank_optimization(31, n, m, p, T_true, min(remaining - 5, 40), rng_seed=2)
        if result_31 is not None:
            U31, V31, W31 = result_31
            score = _compute_residual(U31, V31, W31, n, m, p)
            if score < 1e-6:
                return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}

    # Fall back to rank 32
    U, V, W = _build_rank32()
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
