"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T[i,j,k,l] = delta_{jk}
    T_target = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    T_target[i,j,k,l] = 1.0 if j == k else 0.0

    def compute_tensor(U, V, W):
        """Compute T[i,j,k,l] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+l,r]"""
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T[i,j,k,l] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+l])
        return T

    def try_rank(R, max_restarts=20):
        """Try to find a decomposition of rank R using random initialization and scipy.optimize."""
        from scipy.optimize import minimize
        
        for restart in range(max_restarts):
            np.random.seed(42 + restart)
            
            # Initialize
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # Flatten parameters
            def flatten_params(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten_params(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R:].reshape(p*n, R)
                return U, V, W

            def objective(x):
                U, V, W = unflatten_params(x)
                T = compute_tensor(U, V, W)
                return np.sum((T - T_target)**2)

            x0 = flatten_params(U, V, W)
            
            # Use L-BFGS-B
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success or res.fun < 1e-10:
                U, V, W = unflatten_params(res.x)
                T_final = compute_tensor(U, V, W)
                max_err = np.max(np.abs(T_final - T_target))
                
                if max_err < 1e-6:
                    # Try to quantize to integers/half-integers
                    best_solution = None
                    best_error = float('inf')
                    
                    for scale in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                        scaled_U = U * scale
                        scaled_V = V * scale
                        scaled_W = W * scale
                        
                        # Try integers
                        U_int = np.round(scaled_U)
                        V_int = np.round(scaled_V)
                        W_int = np.round(scaled_W)
                        
                        err_int = max(np.max(np.abs(U_int - scaled_U)), 
                                      np.max(np.abs(V_int - scaled_V)), 
                                      np.max(np.abs(W_int - scaled_W)))
                        
                        if err_int < 1e-4:
                            # Verify
                            T_check = compute_tensor(U_int, V_int, W_int)
                            if np.max(np.abs(T_check - T_target)) < 1e-8:
                                return U_int, V_int, W_int, R

                        # Try half-integers (multiply by 2*scale, round, divide by 2*scale)
                        # Actually, just check if scaled values are close to k/2
                        # This is equivalent to checking if 2*scaled is close to integer
                        U_half = np.round(scaled_U * 2) / 2
                        V_half = np.round(scaled_V * 2) / 2
                        W_half = np.round(scaled_W * 2) / 2
                        
                        err_half = max(np.max(np.abs(U_half - scaled_U)), 
                                       np.max(np.abs(V_half - scaled_V)), 
                                       np.max(np.abs(W_half - scaled_W)))
                        
                        if err_half < 1e-4:
                            T_check = compute_tensor(U_half, V_half, W_half)
                            if np.max(np.abs(T_check - T_target)) < 1e-8:
                                return U_half, V_half, W_half, R

        return None, None, None, R

    # Try ranks from 8 to 12
    for R in range(8, 13):
        U, V, W, rank_used = try_rank(R)
        if U is not None:
            # Convert to list of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: return schoolbook
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
# EVOLVE-BLOCK-END
