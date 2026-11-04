"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Precompute index mappings
    U_rows = n * m  # 15
    V_rows = m * p  # 30
    W_rows = p * n  # 18
    total_eqs = n * m * p  # 90
    
    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def smooth_objective(params, rank):
        """
        Compute the smooth objective: sum of squared residuals.
        params is a flat array of length (n*m + m*p + p*n) * rank
        """
        # Unpack
        U = params[:U_rows*rank].reshape(U_rows, rank)
        V = params[U_rows*rank:(U_rows+V_rows)*rank].reshape(V_rows, rank)
        W = params[(U_rows+V_rows)*rank:].reshape(W_rows, rank)
        
        # Compute residuals
        residual_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    residual_sum += (val - 1.0) ** 2
        return residual_sum
    
    def smooth_grad(params, rank):
        """Compute gradient of the smooth objective."""
        U = params[:U_rows*rank].reshape(U_rows, rank)
        V = params[U_rows*rank:(U_rows+V_rows)*rank].reshape(V_rows, rank)
        W = params[(U_rows+V_rows)*rank:].reshape(W_rows, rank)
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    res = 2.0 * (val - 1.0)
                    # d/dU[i*m+j]
                    dU[i*m+j] += res * (V[j*p+k] * W[k*n+i])
                    # d/dV[j*p+k]
                    dV[j*p+k] += res * (U[i*m+j] * W[k*n+i])
                    # d/dW[k*n+i]
                    dW[k*n+i] += res * (U[i*m+j] * V[j*p+k])
        
        return np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])
    
    def try_rank(rank, max_restarts=10):
        """Try to find a valid decomposition for the given rank."""
        if time.time() > deadline:
            return None
        
        best_result = None
        best_obj = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            params = np.random.randn((U_rows + V_rows + W_rows) * rank) * 0.5
            
            # Use scipy.optimize.minimize with L-BFGS-B
            try:
                from scipy.optimize import minimize
                result = minimize(
                    smooth_objective,
                    params,
                    args=(rank,),
                    jac=smooth_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
            except Exception:
                continue
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
            
            # Check if we found an exact solution
            if result.fun < 1e-10:
                break
        
        if best_result is None or best_obj > 1e-8:
            return None
        
        # Unpack best result
        U = best_result[:U_rows*rank].reshape(U_rows, rank)
        V = best_result[U_rows*rank:(U_rows+V_rows)*rank].reshape(V_rows, rank)
        W = best_result[(U_rows+V_rows)*rank:].reshape(W_rows, rank)
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if verify(U_list, V_list, W_list, rank):
            return U_list, V_list, W_list, rank
        return None
    
    # Start with schoolbook as fallback
    U, V, W, rank = schoolbook()
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, n*p)  # 30
    for target_rank in range(lower_bound, n*m*p + 1):
        if time.time() > deadline:
            break
        
        result = try_rank(target_rank, max_restarts=5)
        if result is not None:
            U, V, W, rank = result
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
