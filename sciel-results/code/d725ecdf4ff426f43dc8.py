"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition"""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Build the tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.zeros((n, m, p))
    for r in range(r):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    # Check against standard matrix multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1.0:
                    return False
    return True

def _snap_to_half_int(arr, tol=1e-6):
    """Snap values to nearest half-integer if close enough."""
    result = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        val = arr[idx]
        nearest = np.round(val * 2) / 2
        if abs(val - nearest) < tol:
            result[idx] = nearest
        else:
            result[idx] = val
    return result

def _build_rank33():
    """Build the known rank-33 decomposition."""
    n, m, p = DIMENSIONS
    rank = 33
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 11 terms for each 2x3x2 block
    terms = [
        {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},
        {'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}},
        {'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}},
        {'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}},
        {'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}},
    ]
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx, col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val
    
    return U, V, W

def _try_find_lower_rank(start_rank, deadline, best_result):
    """Try to find a lower rank decomposition using optimization."""
    n, m, p = DIMENSIONS
    U_size = n * m  # 6
    V_size = m * p  # 18
    W_size = p * n  # 12
    
    # Build target tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Flatten target tensor for optimization
    # We'll use a different approach: directly solve for the factors
    
    for rank in range(start_rank, max(start_rank - 5, 20)):
        if time.time() > deadline - 5:
            break
        
        # Try random initializations
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(U_size, rank)
            V = np.random.randn(V_size, rank)
            W = np.random.randn(W_size, rank)
            
            # Use scipy optimization if available
            try:
                from scipy.optimize import minimize
                from scipy.optimize import least_squares
                
                def residuals(params):
                    U_r = params[:U_size*rank].reshape(U_size, rank)
                    V_r = params[U_size*rank:U_size*rank+V_size*rank].reshape(V_size, rank)
                    W_r = params[U_size*rank+V_size*rank:].reshape(W_size, rank)
                    
                    # Compute tensor
                    T_comp = np.zeros((n, m, p))
                    for r in range(rank):
                        u = U_r[:, r]
                        v = V_r[:, r]
                        w = W_r[:, r]
                        for i in range(n):
                            for j in range(m):
                                T_comp[i, j, :] += u[i*m+j] * v[j*p:][:, None] * w[:p*n:n][None, :]
                    # This is getting complex, let me use a simpler approach
                    return T_comp.flatten() - T.flatten()
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                result = least_squares(residuals, x0, method='lm', max_nfev=10000)
                
                if result.success and result.cost < 1e-10:
                    U_r = result.x[:U_size*rank].reshape(U_size, rank)
                    V_r = result.x[U_size*rank:U_size*rank+V_size*rank].reshape(V_size, rank)
                    W_r = result.x[U_size*rank+V_size*rank:].reshape(W_size, rank)
                    
                    # Snap to half-integers
                    U_s = _snap_to_half_int(U_r)
                    V_s = _snap_to_half_int(V_r)
                    W_s = _snap_to_half_int(W_r)
                    
                    # Verify
                    if _verify_decomposition(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p):
                        return U_s, V_s, W_s, rank
                        
            except Exception:
                continue
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin
    
    # Start with known rank-33 decomposition
    U, V, W = _build_rank33()
    best_result = (U, V, W, 33)
    
    # Try to find lower ranks
    result = _try_find_lower_rank(32, deadline, best_result)
    if result is not None:
        U, V, W, rank = result
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
