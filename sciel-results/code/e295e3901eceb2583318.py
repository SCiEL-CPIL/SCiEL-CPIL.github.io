"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Target tensor: T[i,j,k] = 1 if i == k, else 0
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            T[i, j, i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = delta_{i,k} means only diagonal entries in i,k are 1
    # For n=2, this is a structured tensor
    
    # Try ranks from 49 down to 40
    for target_rank in range(49, 39, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        max_tries = 3
        
        for attempt in range(max_tries):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            scale = 1.0
            U = np.random.randint(-2, 3, size=(n * m, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m * p, target_rank)).astype(float)
            W = np.random.randint(-2, 3, size=(p * n, target_rank)).astype(float)
            
            # Define the residual function
            def residual(params):
                U_flat = params[:n * m * target_rank]
                V_flat = params[n * m * target_rank : n * m * target_rank + m * p * target_rank]
                W_flat = params[n * m * target_rank + m * p * target_rank :]
                
                U_mat = U_flat.reshape(n, m, target_rank)
                V_mat = V_flat.reshape(m, p, target_rank)
                W_mat = W_flat.reshape(p, n, target_rank)
                
                # Compute T_computed[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                T_comp = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_comp[i, j, k] = np.sum(U_mat[i, j, :] * V_mat[j, k, :] * W_mat[k, i, :])
                
                return T_comp - T
            
            # Check initial residual
            res = residual(np.concatenate([U.flatten(), V.flatten(), W.flatten()]))
            if np.max(np.abs(res)) < 1e-8:
                found = True
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
            
            # Try gradient descent / Newton-like method
            # Use scipy's minimize with BFGS
            def objective(params):
                r = residual(params)
                return np.sum(r * r)
            
            def objective_grad(params):
                # Numerical gradient
                eps = 1e-6
                grad = np.zeros_like(params)
                for idx in range(len(params)):
                    params_plus = params.copy()
                    params_plus[idx] += eps
                    params_minus = params.copy()
                    params_minus[idx] -= eps
                    grad[idx] = (objective(params_plus) - objective(params_minus)) / (2 * eps)
                return grad
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(objective, x0, method='BFGS', jac=objective_grad,
                                options={'maxiter': 200, 'gtol': 1e-8})
                
                if result.success or result.fun < 1e-10:
                    x_opt = result.x
                    U_opt = x_opt[:n * m * target_rank].reshape(n, m, target_rank)
                    V_opt = x_opt[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m, p, target_rank)
                    W_opt = x_opt[n * m * target_rank + m * p * target_rank :].reshape(p, n, target_rank)
                    
                    # Verify
                    T_comp = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_comp[i, j, k] = np.sum(U_opt[i, j, :] * V_opt[j, k, :] * W_opt[k, i, :])
                    
                    if np.max(np.abs(T_comp - T)) < 1e-8:
                        # Convert to integer/half-integer if possible
                        U_int = np.round(U_opt)
                        V_int = np.round(V_opt)
                        W_int = np.round(W_opt)
                        
                        # Check if rounded version works
                        T_comp_int = np.zeros((n, m, p))
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T_comp_int[i, j, k] = np.sum(U_int[i, j, :] * V_int[j, k, :] * W_int[k, i, :])
                        
                        if np.max(np.abs(T_comp_int - T)) < 1e-8:
                            U = U_int
                            V = V_int
                            W = W_int
                        else:
                            # Use half-integers
                            U_half = np.round(U_opt * 2) / 2
                            V_half = np.round(V_opt * 2) / 2
                            W_half = np.round(W_opt * 2) / 2
                            
                            T_comp_half = np.zeros((n, m, p))
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        T_comp_half[i, j, k] = np.sum(U_half[i, j, :] * V_half[j, k, :] * W_half[k, i, :])
                            
                            if np.max(np.abs(T_comp_half - T)) < 1e-8:
                                U = U_half
                                V = V_half
                                W = W_half
                            else:
                                U = U_opt
                                V = V_opt
                                W = W_opt
                        
                        found = True
                        best_rank = target_rank
                        best_U = U
                        best_V = V
                        best_W = W
                        break
                        
            except Exception:
                continue
        
        if found:
            break
    
    # Convert to lists for return
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
