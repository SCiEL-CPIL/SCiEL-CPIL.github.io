import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 5]

def _schoolbook():
    """
    Returns the standard schoolbook decomposition for matrix multiplication.
    Rank = n * m * p
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _residuals(x, n, m, p, rank):
    """
    Compute the residual for the tensor decomposition.
    x is a flattened vector of U, V, W.
    """
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    
    U = x[:size_U].reshape(n * m, rank)
    V = x[size_U:size_U + size_V].reshape(m * p, rank)
    W = x[size_U + size_V:].reshape(p * n, rank)
    
    res = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                u_row = U[i * m + j, :]
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                val = np.sum(u_row * v_row * w_row)
                res[idx] = val - 1.0
                idx += 1
    return res

def _round_to_half(x):
    """Round to the nearest half-integer."""
    return np.round(x * 2) / 2.0

def _verify_exact(U, V, W, n, m, p):
    """
    Verify if the decomposition is exact (within floating point tolerance) 
    and if entries are close to half-integers.
    """
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j, :]
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                val = np.sum(u_row * v_row * w_row)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-8:
        return False, max_err
        
    # Check if entries are close to half-integers
    # We allow a small tolerance for the rounding
    U_check = _round_to_half(U)
    V_check = _round_to_half(V)
    W_check = _round_to_half(W)
    
    err_U = np.max(np.abs(U - U_check))
    err_V = np.max(np.abs(V - V_check))
    err_W = np.max(np.abs(W - W_check))
    
    if err_U > 1e-5 or err_V > 1e-5 or err_W > 1e-5:
        return False, max_err
        
    return True, max_err

def _try_reduce_rank(target_rank, time_limit):
    """
    Attempt to find a decomposition of the given target_rank.
    Returns a dict if successful, None otherwise.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # We use multiple random restarts
    # The dimension of the search space is (n*m + m*p + p*n) * rank
    # For 3x5x5, rank 74: (15 + 25 + 15) * 74 = 55 * 74 = 4070 variables.
    # This is a large non-linear least squares problem.
    
    # Strategy: Start from a perturbed schoolbook solution if possible?
    # Schoolbook is rank 75. We can't directly use it for rank 74.
    # Instead, we try random initializations.
    
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        # Scale is important. The entries in schoolbook are 1.
        # Let's try initializing with small random values.
        U_init = np.random.randn(n * m, target_rank) * 0.5
        V_init = np.random.randn(m * p, target_rank) * 0.5
        W_init = np.random.randn(p * n, target_rank) * 0.5
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use 'trf' (trust region reflective) which is good for large problems
            # max_nfev is limited to avoid hanging
            res = least_squares(
                _residuals, 
                x0, 
                args=(n, m, p, target_rank), 
                method='trf', 
                max_nfev=2000,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12
            )
            
            if res.cost < 1e-10:
                # Extract solution
                size_U = n * m * target_rank
                size_V = m * p * target_rank
                U = res.x[:size_U].reshape(n * m, target_rank)
                V = res.x[size_U:size_U + size_V].reshape(m * p, target_rank)
                W = res.x[size_U + size_V:].reshape(p * n, target_rank)
                
                # Check if it's a valid half-integer solution
                is_valid, err = _verify_exact(U, V, W, n, m, p)
                if is_valid:
                    # Convert to lists for the required output format
                    U_list = _round_to_half(U).tolist()
                    V_list = _round_to_half(V).tolist()
                    W_list = _round_to_half(W).tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    total_time_limit = 110 # Leave 10s margin for safety
    
    start_time = time.time()
    
    # First, get the guaranteed schoolbook solution
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find a lower rank solution
    # We try rank 74, then 73, etc., until time runs out
    # Given the difficulty, we focus on rank 74.
    
    remaining_time = total_time_limit - (time.time() - start_time)
    
    # Try rank 74
    if remaining_time > 5:
        result_74 = _try_reduce_rank(74, remaining_time - 5)
        if result_74 is not None:
            best_result = result_74
            best_rank = 74
    
    # If we still have time, try rank 73? 
    # Unlikely to succeed, but let's check time.
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 10 and best_rank > 73:
         result_73 = _try_reduce_rank(73, remaining_time - 5)
         if result_73 is not None:
             best_result = result_73
             best_rank = 73
             
    # Convert numpy arrays to lists if they are still numpy arrays (from schoolbook)
    if isinstance(best_result["U"], np.ndarray):
        best_result["U"] = best_result["U"].tolist()
        best_result["V"] = best_result["V"].tolist()
        best_result["W"] = best_result["W"].tolist()
        
    return best_result
