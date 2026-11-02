import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T

def _compute_error_int(U, V, W, T):
    """
    Compute the sum of squared errors using integer arithmetic.
    U, V, W are integer arrays (scaled by 2).
    T is integer array (target values, 1 or 0).
    Note: The reconstruction is sum_r U[:,r] V[:,r] W[:,r].
    If U,V,W are scaled by 2, the result is scaled by 8.
    So we compare (recon/8) vs T.
    Or equivalently: recon_int vs 8*T.
    """
    # U: (N, R), V: (M, R), W: (P, R)
    # recon[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    target_scaled = 8 * T
    diff = recon - target_scaled
    return np.sum(diff * diff)

def _local_search_int(U, V, W, T, values, max_iter=2000, timeout=5.0):
    """
    Discrete local search using integer arithmetic.
    U, V, W are integer arrays.
    values are the allowed integer values (scaled by 2).
    """
    start_time = time.time()
    current_error = _compute_error_int(U, V, W, T)
    
    n_u, n_v, n_w = U.size, V.size, W.size
    total_vars = n_u + n_v + n_w
    
    # Precompute shapes for reshaping
    shape_U = U.shape
    shape_V = V.shape
    shape_W = W.shape
    
    for _ in range(max_iter):
        if time.time() - start_time > timeout:
            break
        if current_error == 0:
            break
            
        # Pick a random variable
        var_idx = random.randint(0, total_vars - 1)
        
        if var_idx < n_u:
            mat = U
            idx = var_idx
            shape = shape_U
        elif var_idx < n_u + n_v:
            mat = V
            idx = var_idx - n_u
            shape = shape_V
        else:
            mat = W
            idx = var_idx - n_u - n_v
            shape = shape_W
            
        old_val = mat.flat[idx]
        
        # Try all possible new values? Or just one random?
        # Trying all is "Best Improvement". For 9 values, it's cheap.
        best_new_val = old_val
        best_new_error = current_error
        
        for new_val in values:
            if new_val == old_val:
                continue
            mat.flat[idx] = new_val
            new_error = _compute_error_int(U, V, W, T)
            if new_error < best_new_error:
                best_new_error = new_error
                best_new_val = new_val
        
        # Apply the best change
        if best_new_val != old_val:
            mat.flat[idx] = best_new_val
            current_error = best_new_error
        else:
            # If no improvement found, revert if we changed it (we did change it in loop)
            # Actually, if best_new_val is old_val, we just left it as old_val 
            # because we only updated if new_error < best_new_error which starts at current_error.
            # But we did set mat.flat[idx] = new_val in the loop.
            # So we must revert to old_val if no improvement was found.
            mat.flat[idx] = old_val

    return current_error

def _try_rank(R, n, m, p, T, values, time_limit):
    """
    Try to find a decomposition of rank R using discrete search.
    """
    start_time = time.time()
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Number of restarts based on time
    # Each restart is relatively fast.
    num_restarts = 20
    time_per_restart = time_limit / num_restarts
    
    # Values are half-integers, so scaled by 2 they are integers.
    # Original values: [-2.0, -1.5, ..., 2.0]
    # Scaled: [-4, -3, ..., 4]
    int_values = [int(v * 2) for v in values]
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random integers from allowed set
        # Sparse initialization might be better?
        # Let's try random dense first.
        U = np.random.choice(int_values, size=(n * m, R))
        V = np.random.choice(int_values, size=(m * p, R))
        W = np.random.choice(int_values, size=(p * n, R))
        
        # Run local search
        err = _local_search_int(U, V, W, T, int_values, max_iter=1000, timeout=time_per_restart)
        
        if err < best_error:
            best_error = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if best_error == 0:
            break
            
    if best_error == 0:
        # Convert back to floats (divide by 2)
        return best_U / 2.0, best_V / 2.0, best_W / 2.0
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Define search space
    # Integers and half-integers
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Try ranks lower than schoolbook
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    for R in range(15, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
            
        # Time budget for this rank
        ranks_left = rank_schoolbook - R
        time_budget = (deadline - (time.time() - start_time)) / ranks_left
        time_budget = min(time_budget, 20.0)
        
        if time_budget < 1.0:
            break
            
        result = _try_rank(R, n, m, p, T, values, time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify with original float logic to be safe
            # Reconstruct
            recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            # T is int, recon is float.
            # Check if close to T
            if np.allclose(recon, T, atol=1e-9):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
                
    return best_result
