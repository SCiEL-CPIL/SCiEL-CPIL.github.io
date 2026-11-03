"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m + j]
                v_row = V[j*p + k]
                w_row = W[k*n + i]
                s = np.dot(u_row, v_row * w_row)
                err += (s - 1.0) ** 2
    return err

def _snap_and_verify(U, V, W, n, m, p, tol=1e-9):
    """Try to snap to half-integers and verify. Returns dict if valid, else None."""
    U_h = np.round(U * 2) / 2
    V_h = np.round(V * 2) / 2
    W_h = np.round(W * 2) / 2
    
    err_h = _compute_error(U_h, V_h, W_h, n, m, p)
    if err_h < tol:
        return {
            "U": U_h.tolist(),
            "V": V_h.tolist(),
            "W": W_h.tolist()
        }
    
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    
    err_i = _compute_error(U_i, V_i, W_i, n, m, p)
    if err_i < tol:
        return {
            "U": U_i.tolist(),
            "V": V_i.tolist(),
            "W": W_i.tolist()
        }
    
    return None

def _gradient_descent_adaptive(U, V, W, n, m, p, steps=500, lr_init=0.05, lr_decay=0.999, time_limit=10.0):
    """Perform adaptive gradient descent to minimize the error."""
    start = time.time()
    R = U.shape[1]
    lr = lr_init
    
    for step in range(steps):
        if time.time() > start + time_limit:
            break
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                u_row = U[i*m + j]
                for k in range(p):
                    v_row = V[j*p + k]
                    w_row = W[k*n + i]
                    
                    s = np.dot(u_row, v_row * w_row)
                    err_term = 2 * (s - 1.0)
                    
                    dU[i*m + j] += err_term * v_row * w_row
                    dV[j*p + k] += err_term * u_row * w_row
                    dW[k*n + i] += err_term * u_row * v_row
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        lr *= lr_decay
        
    return U, V, W

def _try_rank_with_grad(n, m, p, target_rank, time_limit, num_tries=8):
    """Try to find a decomposition with target_rank using gradient descent."""
    start_time = time.time()
    
    for trial in range(num_tries):
        if time.time() > start_time + time_limit:
            break
            
        scale = 0.05 + 0.1 * random.random()
        U = np.random.randn(n * m, target_rank) * scale
        V = np.random.randn(m * p, target_rank) * scale
        W = np.random.randn(p * n, target_rank) * scale
        
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break
            
        U, V, W = _gradient_descent_adaptive(U, V, W, n, m, p, steps=200, lr_init=0.05, time_limit=remaining * 0.7)
        
        err = _compute_error(U, V, W, n, m, p)
        
        if err < 1e-3:
            result = _snap_and_verify(U, V, W, n, m, p, tol=1e-8)
            if result is not None:
                return result
                
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p
    
    best_result = None
    best_rank = schoolbook_rank
    
    for target_rank in range(74, 24, -1):
        if time.time() > deadline - 15.0:
            break
        
        remaining = deadline - time.time() - 5.0
        if remaining < 2.0:
            break
        
        time_per_method = remaining * 0.9
        result = _try_rank_with_grad(n, m, p, target_rank, time_per_method, num_tries=3)
        
        if result is not None:
            best_result = result
            best_rank = target_rank
        else:
            break
    
    if best_result is not None:
        return best_result
    
    return schoolbook

# EVOLVE-BLOCK-END
