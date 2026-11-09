import numpy as np
import time
from itertools import product
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 135.0  # 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Target: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Schoolbook solution
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
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Verify schoolbook solution
    def verify_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        
        # Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
        
        # The target tensor for matrix multiplication is all ones
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)
    
    # Try to find lower rank decompositions
    # Strategy: Use the structure of the problem
    # For 3x4x7, we can try to find rank < 84
    
    # First, let's try a random search with refinement
    np.random.seed(42)
    
    def objective(params, rank):
        """Compute the error for a given rank decomposition."""
        n, m, p = DIMENSIONS
        total = n * m + m * p + p * n
        U_flat = params[:n * m * rank]
        V_flat = params[n * m * rank:n * m * rank + m * p * rank]
        W_flat = params[n * m * rank + m * p * rank:]
        
        U_arr = U_flat.reshape(n * m, rank)
        V_arr = V_flat.reshape(m * p, rank)
        W_arr = W_flat.reshape(p * n, rank)
        
        # Compute error: sum of (T[i,j,k] - 1)^2
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                    error += (val - 1.0) ** 2
        return error
    
    # Try ranks from 83 down to 60
    for target_rank in range(rank_school - 1, 55, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 25:
                break
            
            # Random initialization with small values
            params_init = np.random.randn(n * m * target_rank + m * p * target_rank + p * n * target_rank) * 0.1
            
            # Use L-BFGS-B optimization
            result = minimize(
                objective,
                params_init,
                args=(target_rank,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e-8:
                # Extract the solution
                U_flat = result.x[:n * m * target_rank]
                V_flat = result.x[n * m * target_rank:n * m * target_rank + m * p * target_rank]
                W_flat = result.x[n * m * target_rank + m * p * target_rank:]
                
                U_sol = U_flat.reshape(n * m, target_rank)
                V_sol = V_flat.reshape(m * p, target_rank)
                W_sol = W_flat.reshape(p * n, target_rank)
                
                # Verify
                if verify_decomposition(U_sol, V_sol, W_sol, tol=1e-6):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    break
        
        if best_rank < rank_school:
            break
    
    # If we found a better solution, use it; otherwise use schoolbook
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
