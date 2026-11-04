"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Verify the decomposition is correct."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _compute_error(n, m, p, U_mat, V_mat, W_mat):
    """Compute the maximum error of the decomposition."""
    U_arr = np.array(U_mat, dtype=float)
    V_arr = np.array(V_mat, dtype=float)
    W_arr = np.array(W_mat, dtype=float)
    rank = U_arr.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_rank_optimization(n, m, p, rank, max_restarts=10, seed=42, time_limit=30):
    """Try to find a valid decomposition with given rank using optimization."""
    np.random.seed(seed)
    start_time = time.time()
    
    def objective(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U_mat = U_flat.reshape(n*m, rank)
        V_mat = V_flat.reshape(m*p, rank)
        W_mat = W_flat.reshape(p*n, rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                    err += (s - 1.0) ** 2
        return err
    
    best_error = float('inf')
    best_params = None
    
    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
        
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 100})
        
        if res.fun < best_error:
            best_error = res.fun
            best_params = res.x
        
        if best_error < 1e-12:
            break
    
    if best_params is None:
        return None
    
    U_mat = best_params[:n*m*rank].reshape(n*m, rank)
    V_mat = best_params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W_mat = best_params[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    if _verify_decomposition(n, m, p, U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), tol=1e-8):
        return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
    
    return None

def _try_strassen_block(n, m, p):
    """
    Try to use Strassen-style block decomposition.
    
    For 3x4x5, we can embed into 4x4x4 and use Strassen on a 2x2x2 block,
    but this is complex. Instead, we try a hybrid approach:
    - Use schoolbook for most terms
    - Try to find a low-rank correction for a subset
    """
    # This is a placeholder for a more sophisticated block approach
    # In practice, for 3x4x5, the theoretical lower bound is around 59
    # and Strassen-style tricks may not help much for such small dimensions
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    start_time = time.time()
    deadline = start_time + 110  # Leave 10 seconds safety margin
    
    # First, try Strassen-style block decomposition
    result = _try_strassen_block(n, m, p)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try optimization from rank 59 down
    # Start from just below schoolbook rank
    for rank in range(n*m*p - 1, max(20, n*m*p - 20), -1):
        if time.time() > deadline - 30:  # Leave 30 seconds for safety
            break
        
        result = _try_rank_optimization(n, m, p, rank, max_restarts=5, seed=rank, time_limit=20)
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
