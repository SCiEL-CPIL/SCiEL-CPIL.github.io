import time
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

DIMENSIONS = [2, 5, 6]
DEADLINE = 105.0  # Safety margin before 110s or 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build target tensor T[i,j,k] = 1 if i,j,k valid indices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)  
    # W shape: (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Start with the known rank-54 solution
    best_U, best_V, best_W = _hardcoded_rank54(n, m, p)
    best_rank = 54
    
    def check_exact(U, V, W, tol=1e-10):
        """Check if decomposition exactly reproduces tensor multiplication."""
        # Compute T_calc[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Target: T_target[i,j,k] = 1
        T_calc = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    T_calc[i, j, k] = s
        # Check if all entries are close to 1
        return np.allclose(T_calc, 1.0, atol=tol)
    
    def round_to_halfint(arr):
        """Round array to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    def try_reduce_rank(U0, V0, W0, target_rank, max_iter=2000):
        """Try to reduce to target_rank using gradient-based optimization."""
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        current_rank = U.shape[1]
        
        if current_rank <= target_rank:
            return U, V, W, current_rank
        
        # Strategy: iteratively zero out columns by optimizing remaining
        # First, try direct optimization with target_rank variables
        
        def objective(params):
            U_r = params[0:target_rank*n*m].reshape(target_rank, n*m).T
            V_r = params[target_rank*n*m:target_rank*(n*m+m*p)].reshape(target_rank, m*p).T
            W_r = params[target_rank*(n*m+m*p):].reshape(target_rank, p*n).T
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                        err += (s - 1.0)**2
            return err
        
        # Initialize with truncated matrices
        U_init = U0[:, :target_rank].flatten()
        V_init = V0[:, :target_rank].flatten()
        W_init = W0[:, :target_rank].flatten()
        params_init = np.concatenate([U_init, V_init, W_init])
        
        try:
            res = minimize(objective, params_init, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < 1e-8:
                U_new = res.x[0:target_rank*n*m].reshape(target_rank, n*m).T
                V_new = res.x[target_rank*n*m:target_rank*(n*m+m*p)].reshape(target_rank, m*p).T
                W_new = res.x[target_rank*(n*m+m*p):].reshape(target_rank, p*n).T
                # Round to half-integers
                U_new = round_to_halfint(U_new)
                V_new = round_to_halfint(V_new)
                W_new = round_to_halfint(W_new)
                if check_exact(U_new, V_new, W_new, tol=1e-8):
                    return U_new, V_new, W_new, target_rank
                # Try without rounding
                if check_exact(U_new, V_new, W_new, tol=1e-8):
                    return U_new, V_new, W_new, target_rank
        except Exception:
            pass
        
        return U0, V0, W0, current_rank
    
    # Try to reduce rank from 54 down
    for target_rank in range(53, max(40, best_rank - 10), -1):
        if time.time() - start_time > DEADLINE:
            break
        
        U_try, V_try, W_try, achieved_rank = try_reduce_rank(
            best_U, best_V, best_W, target_rank, max_iter=500
        )
        
        if achieved_rank < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = achieved_rank
    
    # Try additional reduction with more iterations
    for target_rank in range(best_rank - 1, max(38, best_rank - 5), -1):
        if time.time() - start_time > DEADLINE:
            break
        
        U_try, V_try, W_try, achieved_rank = try_reduce_rank(
            best_U, best_V, best_W, target_rank, max_iter=1000
        )
        
        if achieved_rank < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = achieved_rank
    
    # Final polish: try to clean up entries
    if best_rank < 54:
        # Try rounding again with tighter tolerance
        U_rounded = round_to_halfint(best_U)
        V_rounded = round_to_halfint(best_V)
        W_rounded = round_to_halfint(best_W)
        if check_exact(U_rounded, V_rounded, W_rounded, tol=1e-8):
            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _hardcoded_rank54(n, m, p):
    """Build the known rank-54 Strassen-based decomposition."""
    rank_new = 54
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    
    current_rank = [0]
    
    def add_strassen_block(i_off, j_off, k_off):
        r = current_rank[0]
        # M1
        U_new[i_off*m + j_off, r] = 1
        U_new[(i_off+1)*m + j_off+1, r] = 1
        V_new[j_off*p + k_off, r] = 1
        V_new[(j_off+1)*p + k_off+1, r] = 1
        W_new[k_off*n + i_off, r] = 1
        W_new[(k_off+1)*n + i_off+1, r] = 1
        current_rank[0] += 1
        # M2
        U_new[(i_off+1)*m + j_off, r] = 1
        U_new[(i_off+1)*m + j_off+1, r] = 1
        V_new[j_off*p + k_off, r] = 1
        W_new[k_off*n + i_off+1, r] = 1
        W_new[(k_off+1)*n + i_off+1, r] = -1
        current_rank[0] += 1
        # M3
        U_new[i_off*m + j_off, r] = 1
        V_new[j_off*p + k_off+1, r] = 1
        V_new[(j_off+1)*p + k_off+1, r] = -1
        W_new[(k_off+1)*n + i_off, r] = 1
        W_new[(k_off+1)*n + i_off+1, r] = 1
        current_rank[0] += 1
        # M4
        U_new[(i_off+1)*m + j_off+1, r] = 1
        V_new[(j_off+1)*p + k_off, r] = 1
        V_new[j_off*p + k_off, r] = -1
        W_new[k_off*n + i_off, r] = 1
        W_new[k_off*n + i_off+1, r] = 1
        current_rank[0] += 1
        # M5
        U_new[i_off*m + j_off, r] = 1
        U_new[i_off*m + j_off+1, r] = 1
        V_new[(j_off+1)*p + k_off+1, r] = 1
        W_new[k_off*n + i_off, r] = -1
        W_new[(k_off+1)*n + i_off, r] = 1
        current_rank[0] += 1
        # M6
        U_new[(i_off+1)*m + j_off, r] = 1
        U_new[i_off*m + j_off, r] = -1
        V_new[j_off*p + k_off, r] = 1
        V_new[j_off*p + k_off+1, r] = 1
        W_new[(k_off+1)*n + i_off+1, r] = 1
        current_rank[0] += 1
        # M7
        U_new[i_off*m + j_off+1, r] = 1
        U_new[(i_off+1)*m + j_off+1, r] = -1
        V_new[(j_off+1)*p + k_off, r] = 1
        V_new[(j_off+1)*p + k_off+1, r] = 1
        W_new[k_off*n + i_off, r] = 1
        current_rank[0] += 1
    
    def add_standard_block(i_off, j_off, k_off):
        r = current_rank[0]
        U_new[i_off*m + j_off, r] = 1
        V_new[j_off*p + k_off, r] = 1
        W_new[k_off*n + i_off, r] = 1
        current_rank[0] += 1
        U_new[(i_off+1)*m + j_off, r] = 1
        V_new[j_off*p + k_off, r] = 1
        W_new[k_off*n + i_off+1, r] = 1
        current_rank[0] += 1
    
    # Part 1: A[:, 0:2] * B[0:2, :]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Part 2a: A[:, 2:3] * B[2:3, :]
    for k in range(6):
        add_standard_block(0, 2, k)
        
    # Part 2b: A[:, 3:5] * B[3:5, :]
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    # Trim to actual rank used
    actual_rank = current_rank[0]
    U_new = U_new[:, :actual_rank]
    V_new = V_new[:, :actual_rank]
    W_new = W_new[:, :actual_rank]
    
    return U_new, V_new, W_new
