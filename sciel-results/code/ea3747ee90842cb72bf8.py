"""Program for matrix multiplication tensor decomposition of (3,3,8)."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0


def _schoolbook_decomposition(n, m, p):
    """Build the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor as a 3D array.
    
    T[a, b, c] where a = i*m+j, b = j'*p+k', c = k''*n+i''
    T[a,b,c] = 1 if j=j' and k'=k'' and i''=i, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all diagonal entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal entries (should be 0)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for kp in range(p):
                    for ip in range(n):
                        if jp == j and kp == k and ip == i:
                            continue
                        val = np.sum(U_arr[i*m+j, :] * V_arr[jp*p+kp, :] * W_arr[kp*n+ip, :])
                        if abs(val) > tol:
                            return False
    return True


def _greedy_rank_search(T, n, m, p, max_rank, min_rank, time_budget=30.0):
    """
    Greedy search for a low-rank decomposition using SVD initialization + ALS refinement.
    """
    start_time = time.time()
    
    # Build the tensor as a 2D matrix for SVD
    T_matrix = T.reshape(n * m, m * p * p * n)
    
    # Try various rank values from low to high
    for target_rank in range(min_rank, max_rank + 1):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        try:
            # Use truncated SVD for initialization
            U_svd, s_vals, Vt_svd = np.linalg.svd(T_matrix, full_matrices=False)
            
            if len(s_vals) < target_rank:
                continue
            
            # Truncate to target_rank
            U_init = U_svd[:, :target_rank]
            
            # Initialize V and W randomly with small values
            rng = np.random.RandomState(42 + target_rank)
            V_curr = rng.randn(m * p, target_rank) * 0.1
            W_curr = rng.randn(p * n, target_rank) * 0.1
            U_curr = U_init.copy()
            
            # ALS refinement
            for it in range(40):
                elapsed = time.time() - start_time
                if elapsed > time_budget:
                    break
                
                # Update U
                for a in range(n * m):
                    target = T[a, :, :].flatten()
                    design = (V_curr[:, np.newaxis, :] * W_curr[np.newaxis, :, :]).reshape(m * p * p * n, target_rank)
                    U_curr[a, :] = np.linalg.lstsq(design, target, rcond=None)[0]
                
                # Update V
                for b in range(m * p):
                    target = T[:, b, :].flatten()
                    design = (U_curr[:, np.newaxis, :] * W_curr[np.newaxis, :, :]).reshape(n * m * p * n, target_rank)
                    V_curr[b, :] = np.linalg.lstsq(design, target, rcond=None)[0]
                
                # Update W
                for c in range(p * n):
                    target = T[:, :, c].flatten()
                    design = (U_curr[:, np.newaxis, :] * V_curr[np.newaxis, :, :]).reshape(n * m * m * p, target_rank)
                    W_curr[c, :] = np.linalg.lstsq(design, target, rcond=None)[0]
            
            # Verify
            U_list = U_curr.tolist()
            V_list = V_curr.tolist()
            W_list = W_curr.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
                return U_list, V_list, W_list, target_rank
        
        except Exception:
            continue
    
    return None, None, None, None


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    remaining = DEADLINE_SECONDS - SAFETY_MARGIN - (time.time() - start_time)
    
    if remaining < 20.0:
        return best_result
    
    # Try greedy rank search from moderate ranks downward
    time_budget = min(remaining * 0.6, 50.0)
    U_new, V_new, W_new, rank_new = _greedy_rank_search(T, n, m, p, 
                                                         max_rank=50, 
                                                         min_rank=25,
                                                         time_budget=time_budget)
    
    if U_new is not None and rank_new < best_rank:
        best_result = {"U": U_new, "V": V_new, "W": W_new}
        best_rank = rank_new
    
    remaining = DEADLINE_SECONDS - SAFETY_MARGIN - (time.time() - start_time)
    
    if remaining < 15.0:
        return best_result
    
    # Try specific ranks with fresh starts
    for target_rank in [35, 32, 30, 28, 26]:
        remaining = DEADLINE_SECONDS - SAFETY_MARGIN - (time.time() - start_time)
        if remaining < 20.0:
            break
        
        time_budget = min(remaining * 0.35, 25.0)
        U_try, V_try, W_try, rank_try = _greedy_rank_search(T, n, m, p,
                                                             max_rank=target_rank,
                                                             min_rank=target_rank,
                                                             time_budget=time_budget)
        
        if U_try is not None and rank_try < best_rank:
            best_result = {"U": U_try, "V": V_try, "W": W_try}
            best_rank = rank_try
    
    return best_result
