"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds margin before 120s limit
    
    # Start with schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
    
    # Try to reduce rank via local search
    # We'll try to find a lower-rank decomposition by iteratively removing columns
    # and adjusting the remaining columns
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank
    
    # Strategy: try to find a decomposition with rank < schoolbook_rank
    # by using random initialization and gradient descent
    
    for target_rank in range(schoolbook_rank - 1, max(20, schoolbook_rank // 2), -1):
        if time.time() > deadline - 30:
            break
            
        # Try multiple random initializations
        for trial in range(5):
            if time.time() > deadline - 30:
                break
                
            # Initialize with random values
            U_trial = np.random.randn(n * m, target_rank) * 0.1
            V_trial = np.random.randn(m * p, target_rank) * 0.1
            W_trial = np.random.randn(p * n, target_rank) * 0.1
            
            # Define objective: sum of squared errors over all matrix multiplication entries
            def objective(params):
                # Unpack parameters
                U_flat = params[:n * m * target_rank].reshape(n * m, target_rank)
                V_flat = params[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
                W_flat = params[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
                
                # Compute the tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # We need this to match the schoolbook tensor where T[i,j,k] = 1 for all i,j,k
                # Actually, the tensor for matrix multiplication is T[i,j,k] = delta_{i,j,k} 
                # where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
                # So T[i,j,k] = 1 if the indices are valid, 0 otherwise
                # Wait, let me reconsider. The tensor T satisfies:
                # C[i,k] = sum_{j} sum_{c} U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
                # No, that's not right either.
                
                # The correct formulation: For matrix multiplication C = A * B where A is n x m and B is m x p
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor decomposition gives: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
                # Wait, that's not right. Let me think again.
                
                # The standard tensor decomposition for matrix multiplication:
                # C[i,k] = sum_j A[i,j] B[j,k]
                # = sum_j sum_c U[i*m+j, c] A[i,j] * V[j*p+k, c] B[j,k] * W[k*n+i, c]
                # No, that's still not right.
                
                # Actually, the tensor T is defined such that:
                # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j] B[j,k])
                # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                # So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U_trial[i * m + j, c] * V_trial[j * p + k, c] * W_trial[k * n + i, c]
                            error += (val - 1.0) ** 2
                return error
            
            # Use L-BFGS-B for optimization
            x0 = np.concatenate([U_trial.flatten(), V_trial.flatten(), W_trial.flatten()])
            
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 1000})
            
            if result.success or result.fun < 1e-6:
                U_opt = result.x[:n * m * target_rank].reshape(n * m, target_rank)
                V_opt = result.x[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
                W_opt = result.x[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
                
                if result.fun < 1e-8:
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = target_rank
                    break  # Found a valid decomposition with this rank, try lower
    
    # Convert to lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
