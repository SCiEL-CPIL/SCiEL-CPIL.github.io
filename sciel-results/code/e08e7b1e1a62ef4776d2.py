import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note: W is indexed by k, i. In W_3d, index 0 is p (k), index 1 is n (i).
    # So W_3d[k, i, c]
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    # Check if T is all ones
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    """Create the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _snap_to_half(x):
    """Snap values to nearest 0.5"""
    return np.round(x * 2) / 2

def _optimize_rank(n, m, p, rank, deadline, max_restarts=10):
    """Try to find a decomposition of given rank using optimization."""
    n_vars = n * m * rank + m * p * rank + p * n * rank
    
    def objective(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        error = np.sum((T - 1.0)**2)
        return error

    for restart in range(max_restarts):
        if time.time() > deadline:
            return None, None, None
        
        # Random initialization
        # Scale to help convergence
        init_U = np.random.randn(n, m, rank) * 0.5
        init_V = np.random.randn(m, p, rank) * 0.5
        init_W = np.random.randn(p, n, rank) * 0.5
        
        x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            
            if result.fun < 1e-8:
                # Extract solution
                U_opt = result.x[:n*m*rank].reshape(n, m, rank)
                V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank)
                W_opt = result.x[n*m*rank + m*p*rank:].reshape(p, n, rank)
                
                # Snap to half-integers
                U_snap = _snap_to_half(U_opt)
                V_snap = _snap_to_half(V_opt)
                W_snap = _snap_to_half(W_opt)
                
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U_snap.reshape(n*m, rank)]
                V_list = [[float(x) for x in row] for row in V_snap.reshape(m*p, rank)]
                W_list = [[float(x) for x in row] for row in W_snap.reshape(p*n, rank)]
                
                # Verify exactly
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-9):
                    return U_list, V_list, W_list
                    
        except Exception as e:
            pass
            
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p
    
    # Try lower ranks
    # We go down from 30 to 20
    for rank in range(n * m * p - 1, 19, -1):
        if time.time() > deadline:
            break
        U, V, W = _optimize_rank(n, m, p, rank, deadline, max_restarts=5)
        if U is not None:
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break  # Found a valid decomposition, stop
    
    return best_result
