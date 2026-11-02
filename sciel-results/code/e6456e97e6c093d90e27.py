import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140.0
    
    # Build the rank-53 baseline
    baseline = _build_rank53()
    
    best_result = None
    best_rank = 53
    
    # Try ranks from 50 to 52
    for target_rank in [50, 51, 52]:
        if time.time() > deadline - 15:
            break
        
        try:
            result = _try_find_decomposition(target_rank, deadline)
            if result is not None:
                best_result = result
                best_rank = target_rank
                if target_rank == 50:
                    break
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    return baseline

def _build_rank53():
    n, m, p = 2, 2, 15
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # 2x2x1 factors (last column)
    U_sc = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_sc = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_sc = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    col = 0
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        for r in range(7):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_s[i*2 + j][r]
            for j in range(2):
                V[j*p + k0][c] = V_s[j*2 + 0][r]
                V[j*p + k1][c] = V_s[j*2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_s[k*2 + i][r]
        col += 7
        
    k_last = 14
    for r in range(4):
        c = col + r
        for i in range(2):
            for j in range(2):
                U[i*m + j][c] = U_sc[i*2 + j][r]
        for j in range(2):
            V[j*p + k_last][c] = V_sc[j][r]
        for i in range(2):
            W[k_last*n + i][c] = W_sc[i][r]
            
    return {"U": U, "V": V, "W": W}

def _compute_residual_fast(U, V, W):
    """
    Compute the residual tensor efficiently.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = 2, 2, 15
    rank = U.shape[1]
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # For each (i,j,k), compute sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Vectorize over r
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
    
    return T

def _try_find_decomposition(target_rank, deadline):
    n, m, p = 2, 2, 15
    
    # Build target tensor (all ones)
    T_target = np.ones((n, m, p))
    
    # Get baseline
    base = _build_rank53()
    U_full = np.array(base["U"], dtype=float)
    V_full = np.array(base["V"], dtype=float)
    W_full = np.array(base["W"], dtype=float)
    
    rng = np.random.default_rng(42)
    
    # Try multiple initializations
    num_attempts = 5
    for attempt in range(num_attempts):
        if time.time() > deadline - 10:
            break
        
        if attempt == 0:
            # Start from baseline, drop last columns
            U0 = U_full[:, :target_rank].copy()
            V0 = V_full[:, :target_rank].copy()
            W0 = W_full[:, :target_rank].copy()
        elif attempt == 1:
            # Random initialization
            U0 = rng.normal(0, 0.5, (n*m, target_rank))
            V0 = rng.normal(0, 0.5, (m*p, target_rank))
            W0 = rng.normal(0, 0.5, (p*n, target_rank))
        elif attempt == 2:
            # Baseline with noise
            U0 = U_full[:, :target_rank] + rng.normal(0, 0.1, (n*m, target_rank))
            V0 = V_full[:, :target_rank] + rng.normal(0, 0.1, (m*p, target_rank))
            W0 = W_full[:, :target_rank] + rng.normal(0, 0.1, (p*n, target_rank))
        else:
            # Different random seed
            U0 = rng.normal(0, 0.3, (n*m, target_rank))
            V0 = rng.normal(0, 0.3, (m*p, target_rank))
            W0 = rng.normal(0, 0.3, (p*n, target_rank))
        
        def _flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def _unflatten(x):
            U_size = n * m * target_rank
            V_size = m * p * target_rank
            W_size = p * n * target_rank
            U = x[:U_size].reshape(n * m, target_rank)
            V = x[U_size:U_size+V_size].reshape(m * p, target_rank)
            W = x[U_size+V_size:].reshape(p * n, target_rank)
            return U, V, W
        
        def _residual(x):
            U, V, W = _unflatten(x)
            T_computed = _compute_residual_fast(U, V, W)
            return (T_computed - T_target).flatten()
        
        x0 = _flatten(U0, V0, W0)
        
        try:
            result = least_squares(_residual, x0, method='lm', max_nfev=3000)
            
            if np.linalg.norm(result.fun) < 1e-8:
                U_opt, V_opt, W_opt = _unflatten(result.x)
                
                # Round to half-integers
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify
                T_check = _compute_residual_fast(U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(T_check - T_target)) < 1e-10:
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    
                # Try rounding to integers
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                T_check2 = _compute_residual_fast(U_int, V_int, W_int)
                if np.max(np.abs(T_check2 - T_target)) < 1e-10:
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    
        except Exception:
            continue
    
    return None
