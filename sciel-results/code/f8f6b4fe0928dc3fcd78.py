import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the all-ones tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def tensor_error(U_flat, V_flat, W_flat, n, m, p, r):
    """Calculate the squared error of the tensor decomposition."""
    U = U_flat.reshape(n * m, r)
    V = V_flat.reshape(m * p, r)
    W = W_flat.reshape(p * n, r)
    
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err += (val - 1.0) ** 2
    return err

def find_233_decomposition(max_time=30):
    """
    Find a rank-8 (or lower) half-integer decomposition for 2x3 x 3x3.
    Returns (U, V, W) each as numpy arrays with half-integer entries, or None.
    """
    n, m, p = 2, 3, 3
    start_time = time.time()
    
    # Theoretical minimum rank for 2x3x3 is 8.
    # We try rank 8 first.
    for r in range(8, 5, -1):
        if time.time() - start_time > max_time:
            break
        
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > max_time:
                break
            
            np.random.seed(100 + restart)
            
            # Initialize with small random values
            U_init = np.random.randn(n * m, r) * 0.1
            V_init = np.random.randn(m * p, r) * 0.1
            W_init = np.random.randn(p * n, r) * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U_flat = x[:n*m*r]
                V_flat = x[n*m*r : n*m*r + m*p*r]
                W_flat = x[n*m*r + m*p*r :]
                return tensor_error(U_flat, V_flat, W_flat, n, m, p, r)
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
                
                if res.fun < 1e-8:
                    U_sol = res.x[:n*m*r].reshape(n * m, r)
                    V_sol = res.x[n*m*r : n*m*r + m*p*r].reshape(m * p, r)
                    W_sol = res.x[n*m*r + m*p*r :].reshape(p * n, r)
                    
                    # Round to half-integers
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    if verify_decomposition(U_r, V_r, W_r, n, m, p):
                        return U_r, V_r, W_r, r
            except Exception:
                pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 140  # Safety margin (evaluator kills at 150s)
    
    # Pre-compute schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_rank = rank_school
    best_decomp = (U_school, V_school, W_school)
    
    # Strategy: exploit tensor product structure T(2,3,9) = T(2,3,3) ⊗ I(3)
    # Find rank-r decomposition for 2x3x3, then replicate 3 times for 2x3x9
    result_233 = find_233_decomposition(max_time=60)
    
    if result_233 is not None and time.time() - start_time < timeout - 10:
        U233, V233, W233, r233 = result_233
        # Replicate 3 times for the 9-column case
        total_rank = 3 * r233
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # For each block b=0,1,2 (corresponding to columns 3b, 3b+1, 3b+2):
        # U[i*m+j, b*r233 + c] = U233[i*m+j, c]
        # V[j*p+k, b*r233 + c] = V233[j*p + (k % 3), c] if k in block b, else 0
        # W[k*n+i, b*r233 + c] = W233[(k%3)*n + i, c] if k in block b, else 0
        
        for b in range(3):
            offset = b * r233
            U[:, offset:offset + r233] = U233
            
            for j in range(m):
                for k in range(p):
                    block_idx = k // 3
                    local_k = k % 3
                    if block_idx == b:
                        V[j * p + k, offset:offset + r233] = V233[j * 3 + local_k, :]
            
            for k in range(p):
                block_idx = k // 3
                local_k = k % 3
                if block_idx == b:
                    for i in range(n):
                        W[k * n + i, offset:offset + r233] = W233[local_k * n + i, :]
        
        if verify_decomposition(U, V, W, n, m, p):
            best_rank = total_rank
            best_decomp = (U, V, W)
    
    # Return the best found
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
