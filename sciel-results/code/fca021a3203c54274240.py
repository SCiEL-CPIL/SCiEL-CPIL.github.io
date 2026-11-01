import time
import math
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    We use a gradient-based optimizer to find a low-rank decomposition.
    The target rank is 54 (schoolbook), but we attempt to find lower ranks
    by solving the nonlinear system directly.
    
    We return the verified schoolbook decomposition as a fallback.
    """
    n, m, p = 3, 3, 6
    
    # Start with the schoolbook decomposition as a guaranteed valid solution
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    schoolbook = get_schoolbook()
    
    # Verify function
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # If schoolbook is valid, we have a baseline
    if verify(schoolbook["U"], schoolbook["V"], schoolbook["W"], n, m, p):
        best_result = schoolbook
    else:
        return schoolbook
    
    # Try to find a lower-rank decomposition using optimization
    # The tensor entries T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T[i,j,k] = 1 for all i,j,k
    
    # Define the objective function for a given rank
    def make_objective(rank, n, m, p):
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        
        def objective(x):
            U = x[:dim_u * rank].reshape(dim_u, rank)
            V = x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
            W = x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                        error += (s - 1.0) ** 2
            return error
        
        def gradient(x):
            U = x[:dim_u * rank].reshape(dim_u, rank)
            V = x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
            W = x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
            
            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                        err = s - 1.0
                        # d/dU[i*m+j, c] = err * V[j*p+k, c] * W[k*n+i, c]
                        gU[i * m + j] += err * V[j * p + k] * W[k * n + i]
                        # d/dV[j*p+k, c] = err * U[i*m+j, c] * W[k*n+i, c]
                        gV[j * p + k] += err * U[i * m + j] * W[k * n + i]
                        # d/dW[k*n+i, c] = err * U[i*m+j, c] * V[j*p+k, c]
                        gW[k * n + i] += err * U[i * m + j] * V[j * p + k]
            
            grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            return grad
        
        return objective, gradient
    
    # Try ranks from 54 down to 30
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    for target_rank in range(54, 29, -1):
        if time.time() > deadline:
            break
        
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        total_dim = (dim_u + dim_v + dim_w) * target_rank
        
        objective, gradient = make_objective(target_rank, n, m, p)
        
        # Initialize with random sparse values
        np.random.seed(42 + target_rank)
        x0 = np.random.randn(total_dim) * 0.1
        
        # Use L-BFGS-B
        try:
            result = minimize(
                objective, 
                x0, 
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < 1e-8:
                # Extract U, V, W
                U_flat = result.x[:dim_u * target_rank].reshape(dim_u, target_rank)
                V_flat = result.x[dim_u * target_rank:dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                W_flat = result.x[dim_u * target_rank + dim_v * target_rank:].reshape(dim_w, target_rank)
                
                # Convert to lists
                U_list = U_flat.tolist()
                V_list = V_flat.tolist()
                W_list = W_flat.tolist()
                
                # Verify
                if verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            continue
    
    # Return the best valid result found
    return best_result
