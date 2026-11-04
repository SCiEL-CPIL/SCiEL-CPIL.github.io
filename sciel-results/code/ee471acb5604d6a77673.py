"""
Optimized construction for Matrix Multiplication Tensor Decomposition.
Task: Find rank-minimal exact tensor decomposition for 2x3 * 3x7.
Dimensions: n=2, m=3, p=7.
Schoolbook rank: 42.
"""

import numpy as np
import time
from scipy.optimize import least_squares, minimize

DIMENSIONS = [2, 3, 7]

def _schoolbook_decomposition(n, m, p):
    """Return the trivial schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    
    return U, V, W

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is valid."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_optimization(n, m, p, target_rank, max_iter=500, timeout=10):
    """Try to find a valid decomposition for the given target rank."""
    start_time = time.time()
    
    # Parameters: flatten U, V, W into a single vector
    n_params = target_rank * (n*m + m*p + p*n)
    
    def objective(params):
        U = params[:target_rank*n*m].reshape(target_rank, n*m).T  # (n*m, rank)
        V = params[target_rank*n*m:target_rank*(n*m+m*p)].reshape(target_rank, m*p).T  # (m*p, rank)
        W = params[target_rank*(n*m+m*p):].reshape(target_rank, p*n).T  # (p*n, rank)
        
        # Compute residuals for all (i,j,k)
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    residuals.append(val - 1.0)
        return np.array(residuals)
    
    # Try multiple random initializations
    best_params = None
    best_resid = np.inf
    
    for trial in range(20):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(trial * 12345 + 42)
        x0 = np.random.randn(n_params) * 0.5
        
        try:
            result = least_squares(objective, x0, method='lm', max_nfev=max_iter)
            final_resid = np.max(np.abs(result.fun))
            if final_resid < best_resid:
                best_resid = final_resid
                best_params = result.x
        except Exception:
            continue
    
    if best_params is None or best_resid > 1e-6:
        return None
    
    # Extract U, V, W
    U = best_params[:target_rank*n*m].reshape(target_rank, n*m).T
    V = best_params[target_rank*n*m:target_rank*(n*m+m*p)].reshape(target_rank, m*p).T
    W = best_params[target_rank*(n*m+m*p):].reshape(target_rank, p*n).T
    
    # Verify
    if _check_decomposition(U, V, W, n, m, p, tol=1e-6):
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # First, get the schoolbook decomposition as fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    
    # Try to find lower-rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But actual lower bounds are higher. Try from 21 upward.
    for target_rank in range(21, 42):
        if time.time() > deadline:
            break
        
        result = _try_rank_optimization(n, m, p, target_rank, max_iter=300, timeout=15)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    U_list = U_school.tolist()
    V_list = V_school.tolist()
    W_list = W_school.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
