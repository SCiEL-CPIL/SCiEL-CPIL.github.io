import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 9]

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decompose the all-ones tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def polish_half_integer(U, V, W, n, m, p, max_steps=100):
    """
    Attempt to polish a near-half-integer solution into an exact half-integer solution
    using local search on the discrete grid.
    """
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    if verify_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    # Calculate initial error
    def calc_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        return err
    
    current_err = calc_error(U, V, W)
    
    for _ in range(max_steps):
        improved = False
        # Try perturbing U
        for i in range(n):
            for j in range(m):
                for r in range(U.shape[1]):
                    original = U[i * m + j, r]
                    for delta in [-0.5, 0.5]:
                        U[i * m + j, r] = original + delta
                        new_err = calc_error(U, V, W)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                        else:
                            U[i * m + j, r] = original
        # Try perturbing V
        for j in range(m):
            for k in range(p):
                for r in range(V.shape[1]):
                    original = V[j * p + k, r]
                    for delta in [-0.5, 0.5]:
                        V[j * p + k, r] = original + delta
                        new_err = calc_error(U, V, W)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                        else:
                            V[j * p + k, r] = original
        # Try perturbing W
        for k in range(p):
            for i in range(n):
                for r in range(W.shape[1]):
                    original = W[k * n + i, r]
                    for delta in [-0.5, 0.5]:
                        W[k * n + i, r] = original + delta
                        new_err = calc_error(U, V, W)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                        else:
                            W[k * n + i, r] = original
        
        if verify_decomposition(U, V, W, n, m, p):
            return U, V, W
            
        if not improved:
            break
            
    return None

def find_233_decomposition(max_time=30):
    """
    Find a rank-7 (or lower) half-integer decomposition for 2x3 x 3x3.
    Returns (U, V, W) each as numpy arrays with half-integer entries, or None.
    """
    n, m, p = 2, 3, 3
    start_time = time.time()
    
    # Try ranks from 7 down to theoretical minimum
    for r in range(7, 5, -1): # Try 7, 6, 5
        if time.time() - start_time > max_time:
            break
        
        max_restarts = 100
        for restart in range(max_restarts):
            if time.time() - start_time > max_time:
                break
            
            np.random.seed(42 + restart * 7 + r)
            
            # Initialize with half-integers
            U = np.random.randint(-4, 5, (n * m, r)) * 0.5
            V = np.random.randint(-4, 5, (m * p, r)) * 0.5
            W = np.random.randint(-4, 5, (p * n, r)) * 0.5
            
            # Use scipy least_squares for faster convergence
            def objective(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r :].reshape(p*n, r)
                
                residuals = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_flat[i*m+j, :], V_flat[j*p+k, :] * W_flat[k*n+i, :])
                            residuals.append(val - 1.0)
                return residuals
            
            try:
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                res = least_squares(objective, x0, method='lm', max_nfev=1000)
                
                U_opt = res.x[:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Check if close to solution
                if res.cost < 1e-10:
                    # Polish to half-integers
                    result = polish_half_integer(U_opt, V_opt, W_opt, n, m, p, max_steps=50)
                    if result is not None:
                        return result[0], result[1], result[2], r
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
    
    # Strategy: exploit tensor product structure T(2,3,9) = T(2,3,3) ⊕ T(2,3,3) ⊕ T(2,3,3)
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
        # V[j*p+k, b*r233 + c] = V233[j * 3 + (k % 3), c] if k in block b, else 0
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
    
    # If we didn't find a good 233 decomposition, try direct search on 2x3x9
    # This is less likely to succeed with high rank reduction, but we try a few ranks
    if best_rank == rank_school and time.time() - start_time < timeout - 10:
        # Try ranks from 27 down to 24 (corresponding to 233 ranks 9, 8)
        for r in range(27, 23, -1):
            if time.time() - start_time > timeout - 10:
                break
            
            max_restarts = 5
            for restart in range(max_restarts):
                if time.time() - start_time > timeout - 10:
                    break
                
                np.random.seed(5000 + r * 100 + restart)
                
                # Half-integer initialization
                U = np.random.randint(-4, 5, (n * m, r)) * 0.5
                V = np.random.randint(-4, 5, (m * p, r)) * 0.5
                W = np.random.randint(-4, 5, (p * n, r)) * 0.5
                
                # ALS iterations
                for it in range(50):
                    if time.time() - start_time > timeout - 10:
                        break
                    
                    # Update U
                    for i in range(n):
                        for j in range(m):
                            V_sub = V[j * p:(j + 1) * p, :]
                            W_sub = np.array([W[k * n + i, :] for k in range(p)])
                            A = V_sub * W_sub
                            try:
                                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                                U[i * m + j, :] = u
                            except:
                                pass
                    
                    # Update V
                    for j in range(m):
                        for k in range(p):
                            U_sub = np.array([U[i * m + j, :] for i in range(n)])
                            W_sub = W[k * n:(k + 1) * n, :]
                            A = U_sub * W_sub
                            try:
                                v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                                V[j * p + k, :] = v
                            except:
                                pass
                    
                    # Update W
                    for k in range(p):
                        for i in range(n):
                            U_sub = np.array([U[i * m + j, :] for j in range(m)])
                            V_sub = np.array([V[j * p + k, :] for j in range(m)])
                            A = U_sub * V_sub
                            try:
                                w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                                W[k * n + i, :] = w
                            except:
                                pass
                    
                    # Check error
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                                err += (val - 1.0) ** 2
                    
                    if err < 1e-10:
                        break
                
                # Round to half-integers and verify
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify_decomposition(U_r, V_r, W_r, n, m, p):
                    best_rank = r
                    best_decomp = (U_r, V_r, W_r)
                    break
    
    # Return the best found
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
