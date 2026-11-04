"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    rank_school = n * m * p
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Verify a candidate decomposition
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    # Try to find lower rank using optimization
    def try_rank(target_rank, max_iter=200):
        """Try to find a decomposition of given rank using optimization."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(42)
        
        # Initialize with random values
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Define the objective: minimize sum of squared errors
        def objective(params):
            if time.time() > deadline - 3:
                return 1e10
            
            # Reshape parameters
            U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_flat[k*n+i, :])
                        error += (val - 1.0) ** 2
            return error
        
        # Initial parameters
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use Nelder-Mead or L-BFGS-B
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                             options={'maxiter': max_iter, 'maxfun': 5000})
            
            if time.time() > deadline - 3:
                return None
            
            # Extract solution
            U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_sol = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_sol = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            # Check if solution is good
            if result.fun < 1e-10:
                # Verify exactly
                if verify(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), target_rank):
                    return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
            
            # Try with tighter tolerance
            if result.fun < 1e-6:
                # Polish the solution
                for _ in range(50):
                    if time.time() > deadline - 3:
                        break
                    # Compute gradient and do a small step
                    params = result.x
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = (i*m+j)*target_rank
                                idx_v = (n*m + j*p+k)*target_rank
                                idx_w = (n*m + m*p + k*n+i)*target_rank
                                # Skip for now, just check
                    break
                
                U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                if verify(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), target_rank):
                    return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
                    
        except Exception:
            pass
        
        return None
    
    # Try ranks from low to high
    best_U, best_V, best_W = make_schoolbook()
    best_rank = rank_school
    
    # Try ranks from 30 to 59
    for target_rank in range(30, rank_school):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            best_rank = target_rank
            break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        U, V, W = make_schoolbook()
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
