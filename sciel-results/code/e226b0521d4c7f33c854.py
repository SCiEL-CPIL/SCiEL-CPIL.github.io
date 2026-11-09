"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The matrix multiplication tensor T[i,j,k] has entries T[(i,j),(j',k),(i',k')] = delta_{j,j'} delta_{i,i'}
    # A rank-r decomposition means T = sum_{c=1}^r U_c ⊗ V_c ⊗ W_c
    # where U_c is a vector of length n*m, V_c length m*p, W_c length p*n
    
    # We'll use a simple approach: try rank values from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(6, 9, 6) = 9
    # But actually for matrix multiplication, the lower bound is more complex.
    # Known: rank of 2x3 by 3x3 matrix multiplication is at least... let's try values
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j] * T[j',k] * T[i',k'] where the product is delta_{j,j'} delta_{i,i'}
                # So T[i*m+j, j'*p+k, k'*n+i] = 1 if j==j' and i==i'
                # i.e., T[i*m+j, j*p+k, k*n+i] = 1
                T[i*m + j, j*p + k, k*n + i] = 1
    
    # Try different ranks using random initialization and gradient descent
    for rank in range(9, best_rank + 1):
        if time.time() > deadline - 20:
            break
        
        # Multiple random restarts
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Define objective: ||T - sum_c U_c ⊗ V_c ⊗ W_c||_F^2
            def objective(params):
                U_flat, V_flat, W_flat = np.split(params, [n*m*rank, n*m*rank + m*p*rank])
                U_mat = U_flat.reshape(n*m, rank)
                V_mat = V_flat.reshape(m*p, rank)
                W_mat = W_flat.reshape(p*n, rank)
                
                # Compute the tensor sum
                # T_approx[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
                T_approx = np.einsum('ic,jc,kc->ijk', U_mat, V_mat, W_mat)
                diff = T - T_approx
                return np.sum(diff**2)
            
            # Try to minimize
            try:
                params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                result = minimize(objective, params0, method='BFGS', maxiter=50, 
                                options={'gtol': 1e-6, 'disp': False})
                
                if result.fun < 1e-6:
                    # Check if we can clean up to get exact integer/half-integer values
                    U_mat, V_mat, W_mat = np.split(result.x, [n*m*rank, n*m*rank + m*p*rank])
                    U_mat = U_mat.reshape(n*m, rank)
                    V_mat = V_mat.reshape(m*p, rank)
                    W_mat = W_mat.reshape(p*n, rank)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_mat * 2) / 2
                    V_rounded = np.round(V_mat * 2) / 2
                    W_rounded = np.round(W_mat * 2) / 2
                    
                    # Verify the rounded solution
                    T_check = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_check, T, atol=1e-10):
                        # Convert to lists
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        break
            except Exception:
                pass
        
        if best_rank == rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
