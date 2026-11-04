"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _tensor_product(U, V, W, n, m, p):
    """Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    using vectorized operations."""
    rank = U.shape[1]
    # Reshape U, V, W to 3D tensors
    U3 = U.reshape(n, m, rank)  # (n, m, rank)
    V3 = V.reshape(m, p, rank)  # (m, p, rank)
    W3 = W.reshape(p, n, rank)  # (p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum for efficient computation
    # U[i,j,c] * V[j,k,c] * W[k,i,c]
    # First: U[i,j,c] * V[j,k,c] -> (n, m, p, c)
    UV = np.einsum('ijc,jkc->ijkc', U3, V3)  # (n, m, p, rank)
    # Then: UV[i,j,k,c] * W[k,i,c] -> (n, m, p)
    T = np.einsum('ijkc,kic->ijk', UV, W3)  # (n, m, p)
    return T

def _compute_max_error(U, V, W, n, m, p):
    """Compute maximum |T[i,j,k] - 1| for the tensor decomposition."""
    T = _tensor_product(U, V, W, n, m, p)
    return np.max(np.abs(T - 1.0))

def _check_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Check if the decomposition is valid using vectorized computation."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    return _compute_max_error(U_arr, V_arr, W_arr, n, m, p) < tol

def _try_rank_vectorized(n, m, p, rank, max_restarts=20, seed=42, time_limit=30.0):
    """Try to find a valid decomposition with given rank using vectorized operations."""
    start_time = time.time()
    np.random.seed(seed)
    
    best_error = float('inf')
    best_params = None
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random values
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank:n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            U_mat = U_flat.reshape(n*m, rank)
            V_mat = V_flat.reshape(m*p, rank)
            W_mat = W_flat.reshape(p*n, rank)
            
            T = _tensor_product(U_mat, V_mat, W_mat, n, m, p)
            err = np.sum((T - 1.0) ** 2)
            return err
        
        res = minimize(objective, params, method='L-BFGS-B', 
                      options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_error:
            best_error = res.fun
            best_params = res.x
        
        if best_error < 1e-14:
            break
    
    if best_params is None or best_error > 1e-10:
        return None
    
    U_mat = best_params[:n*m*rank].reshape(n*m, rank)
    V_mat = best_params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W_mat = best_params[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    # Check if valid
    if _check_decomposition(n, m, p, U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), tol=1e-8):
        return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    start_time = time.time()
    deadline = start_time + 115  # Leave 5 seconds safety margin
    
    # Try ranks from lower to upper (more likely to find low rank first)
    # Start with reasonable lower bounds
    for rank in range(20, n*m*p):
        if time.time() > deadline - 10:  # Leave 10 seconds for safety
            break
        
        result = _try_rank_vectorized(n, m, p, rank, max_restarts=15, seed=rank+100, 
                                     time_limit=min(20.0, deadline - time.time() - 5))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
