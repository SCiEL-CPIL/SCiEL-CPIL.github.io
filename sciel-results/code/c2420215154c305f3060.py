"""Program for matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # Start with trivial rank-60 decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to reduce rank by checking if we can merge columns
    # For each pair of columns, check if they can be combined
    # This is a heuristic search - try small rank reductions
    
    # Try to find a decomposition with rank < 60
    # Use the fact that we need: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try rank from 59 down to 50
    for target_rank in range(rank - 1, max(rank - 10, 0), -1):
        # Initialize with first target_rank columns of trivial decomposition
        U_t = U[:, :target_rank].copy()
        V_t = V[:, :target_rank].copy()
        W_t = W[:, :target_rank].copy()
        
        # Add some random perturbation to help escape local minima
        np.random.seed(42)
        U_t = U_t + 0.1 * np.random.randn(n * m, target_rank)
        V_t = V_t + 0.1 * np.random.randn(m * p, target_rank)
        W_t = W_t + 0.1 * np.random.randn(p * n, target_rank)
        
        # Use scipy to solve the nonlinear system
        try:
            from scipy.optimize import least_squares
            
            def residuals(params):
                # Unpack parameters
                U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_flat = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Compute tensor entries
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(target_rank):
                                val += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                            res.append(val - 1)
                return np.array(res)
            
            # Initial guess
            x0 = np.concatenate([U_t.flatten(), V_t.flatten(), W_t.flatten()])
            
            result = least_squares(residuals, x0, method='lm', max_nfev=10000)
            
            if result.success or result.cost < 1e-6:
                U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Round to nearest half-integer
                U_sol = np.round(U_sol * 2) / 2
                V_sol = np.round(V_sol * 2) / 2
                W_sol = np.round(W_sol * 2) / 2
                
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(target_rank):
                                val += U_sol[i*m+j, r] * V_sol[j*p+k, r] * W_sol[k*n+i, r]
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    break
        except Exception:
            continue
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
