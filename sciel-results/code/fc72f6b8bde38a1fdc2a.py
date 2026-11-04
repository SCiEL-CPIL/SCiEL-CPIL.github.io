"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Build the target tensor
    # T[(i,j), (j,k), (i,k)] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a low-rank decomposition
    # We'll search for ranks from lower bounds up to schoolbook
    schoolbook_rank = n * m * p
    
    # Lower bound: max dimension of the tensor modes
    lower_bound = max(n*m, m*p, p*n)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Start with schoolbook as fallback
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    best_result = (U_sch, V_sch, W_sch)
    
    # Try to find lower rank decompositions
    # Use a grid search + optimization approach
    for target_rank in range(lower_bound, min(schoolbook_rank, 40)):
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts for this rank
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random or structured values
            if restart == 0:
                # Structured initialization based on schoolbook
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                # Try to map schoolbook terms to fewer columns
                for idx in range(schoolbook_rank):
                    if idx < target_rank:
                        U_init[:, idx] = U_sch[:, idx]
                        V_init[:, idx] = V_sch[:, idx]
                        W_init[:, idx] = W_sch[:, idx]
            else:
                U_init = np.random.randn(n*m, target_rank) * 0.5
                V_init = np.random.randn(m*p, target_rank) * 0.5
                W_init = np.random.randn(p*n, target_rank) * 0.5
            
            # Define objective: minimize ||tensor - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||^2
            def objective(params):
                U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Compute tensor approximation
                approx = np.zeros_like(T)
                for r in range(target_rank):
                    approx += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
                
                return np.sum((T - approx)**2)
            
            params_init = np.concatenate([
                U_init.flatten(),
                V_init.flatten(),
                W_init.flatten()
            ])
            
            try:
                result = minimize(objective, params_init, method='L-BFGS-B', 
                                 options={'maxiter': 200, 'maxfun': 500})
                
                if result.fun < 1e-8:
                    # Found a good decomposition, try to round to half-integers
                    U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U_r * 2) / 2
                    V_rounded = np.round(V_r * 2) / 2
                    W_rounded = np.round(W_r * 2) / 2
                    
                    # Verify
                    approx = np.zeros_like(T)
                    for r in range(target_rank):
                        approx += np.einsum('a,b,c->abc', U_rounded[:,r], V_rounded[:,r], W_rounded[:,r])
                    
                    if np.allclose(T, approx, atol=1e-10):
                        # Convert to lists
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        best_result = (U_list, V_list, W_list)
                        best_rank = target_rank
                        break  # This rank works, try lower
            except Exception:
                pass
        
        if best_rank == target_rank:
            # Found a working rank, continue to try lower
            pass
        else:
            # Could not find this rank, continue
            pass
    
    # Convert best result to the required format
    if isinstance(best_result[0], np.ndarray):
        U_final = best_result[0].tolist()
        V_final = best_result[1].tolist()
        W_final = best_result[2].tolist()
    else:
        U_final = best_result[0]
        V_final = best_result[1]
        W_final = best_result[2]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
