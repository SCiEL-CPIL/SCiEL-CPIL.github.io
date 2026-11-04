"""
Optimized construction for Matrix Multiplication Tensor Decomposition.
Task: Find rank-minimal exact tensor decomposition for 2x3 * 3x7.
Dimensions: n=2, m=3, p=7.
Schoolbook rank: 42.
"""

import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]
DEADLINE = 140.0  # Safety margin before 150s kill

def _build_tensor_target(n, m, p):
    """Build the target tensor flattened as a vector of length n*m*p."""
    # T[i,j,k] = 1 for all valid i,j,k
    # Flattened index: (i*m + j) * p + k for A part, but we need the full tensor
    # The tensor has shape (n*m, m*p, p*n) in the decomposition sense
    # Target: for each (i,j,k), the product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should equal 1
    # So we're solving: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    pass

def _residuals(U, V, W, n, m, p):
    """Compute residuals for the tensor decomposition."""
    rank = U.shape[1]
    # Residual for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1
    res = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                res[idx] = val - 1.0
                idx += 1
    return res

def _try_rank(n, m, p, rank, time_limit):
    """Try to find a decomposition of given rank."""
    start_time = time.time()
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    best_loss = np.inf
    best_params = None
    
    for restart in range(5):
        if time.time() - start_time > time_limit * 0.8:
            break
            
        # Re-initialize for each restart
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        def pack_params():
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unpack_params(params):
            u_size = n * m * rank
            v_size = m * p * rank
            U_p = params[:u_size].reshape(n * m, rank)
            V_p = params[u_size:u_size+v_size].reshape(m * p, rank)
            W_p = params[u_size+v_size:].reshape(p * n, rank)
            return U_p, V_p, W_p
        
        def residual_func(params):
            U_p, V_p, W_p = unpack_params(params)
            return _residuals(U_p, V_p, W_p, n, m, p)
        
        try:
            result = least_squares(
                residual_func, 
                pack_params(), 
                method='lm',
                max_nfev=2000
            )
            loss = np.sum(result.fun**2)
            
            if loss < best_loss:
                best_loss = loss
                best_params = result.x
            
            if time.time() - start_time > time_limit:
                break
                
        except Exception:
            continue
    
    if best_params is None:
        return None, np.inf
    
    U_final, V_final, W_final = unpack_params(best_params)
    
    # Check if solution is close to integer/half-integer
    # Scale to try to get nice numbers
    # First check if current values are close to half-integers
    all_vals = np.concatenate([U_final.flatten(), V_final.flatten(), W_final.flatten()])
    
    # Try to find a scaling factor that makes values close to half-integers
    # For now, check if the solution is valid (residuals small)
    if best_loss < 1e-10:
        # Round to nearest half-integer and verify
        U_round = np.round(U_final * 2) / 2.0
        V_round = np.round(V_final * 2) / 2.0
        W_round = np.round(W_final * 2) / 2.0
        
        res_check = _residuals(U_round, V_round, W_round, n, m, p)
        if np.max(np.abs(res_check)) < 1e-8:
            return (U_round, V_round, W_round), best_loss
    
    return None, best_loss

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, c] = 1.0
                V_school[j * p + k, c] = 1.0
                W_school[k * n + i, c] = 1.0
                c += 1
    
    # Try to find lower rank decompositions
    # Based on Strassen-inspired block decomposition, try ranks from 28 upward
    best_result = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # Try specific ranks that might be achievable
    ranks_to_try = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    
    for r in ranks_to_try:
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time for this rank attempt
        time_for_rank = min(remaining * 0.7, 15.0)
        
        result, loss = _try_rank(n, m, p, r, time_for_rank)
        
        if result is not None:
            best_result = result
            best_rank = r
            break  # Found a valid decomposition, use it
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 2.0:
            break
    
    U_final, V_final, W_final = best_result
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
