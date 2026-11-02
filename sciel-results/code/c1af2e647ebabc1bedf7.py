"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140  # Leave 10s margin
    n, m, p = DIMENSIONS
    
    # Start with the trivial rank-105 decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Try to reduce rank using optimization
    # The tensor T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # We want to find minimal rank r such that this holds
    
    # Try ranks from a lower bound up
    # Lower bound for matrix multiplication is max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But we can try higher values
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find a lower rank decomposition
    for target_rank in range(35, min(rank, 100)):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random values
        np.random.seed(42)
        U_init = np.random.randn(n * m, target_rank) * 0.1
        V_init = np.random.randn(m * p, target_rank) * 0.1
        W_init = np.random.randn(p * n, target_rank) * 0.1
        
        # Flatten parameters
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(params):
            U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            # Compute the tensor: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
            # Compare with target tensor where T_{i,j,k} = 1 for all i,j,k
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_flat[k*n+i, :])
                        error += (val - 1.0) ** 2
            return error
        
        # Try a few iterations of optimization
        try:
            result = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6})
            if result.fun < 1e-8:
                U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Verify
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_opt[i*m+j, :] * V_opt[j*p+k, :] * W_opt[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = target_rank
                    break
        except:
            pass
    
    # Convert to lists of lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
