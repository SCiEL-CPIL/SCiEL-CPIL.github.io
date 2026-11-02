"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def loss(params, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:size_U + size_V + size_W]
        
        U = U_flat.reshape(n * m, R)
        V = V_flat.reshape(m * p, R)
        W = W_flat.reshape(p * n, R)
        
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T[i, j, k, l] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+l])
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        target = 1.0 if j == k else 0.0
                        error += (T[i, j, k, l] - target) ** 2
        
        return error

    def try_rank(R):
        best_result = None
        best_loss = float('inf')
        
        for restart in range(20):
            np.random.seed(restart)
            U_init = np.random.randn(n * m, R)
            V_init = np.random.randn(m * p, R)
            W_init = np.random.randn(p * n, R)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            res = minimize(loss, params_init, args=(R,), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_result = res
        
        if best_result is None or best_loss > 1e-6:
            return None
            
        params = best_result.x
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U = params[:size_U].reshape(n * m, R)
        V = params[size_U:size_U + size_V].reshape(m * p, R)
        W = params[size_U + size_V:size_U + size_V + size_W].reshape(p * n, R)
        
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        
        T_check = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T_check[i, j, k, l] = np.dot(U_int[i*m+j], V_int[j*p+k] * W_int[k*n+l])
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        target = 1.0 if j == k else 0.0
                        max_err = max(max_err, abs(T_check[i, j, k, l] - target))
        
        if max_err < 1e-8:
            return {
                "U": U_int.tolist(),
                "V": V_int.tolist(),
                "W": W_int.tolist()
            }
            
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        T_check = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T_check[i, j, k, l] = np.dot(U_half[i*m+j], V_half[j*p+k] * W_half[k*n+l])
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        target = 1.0 if j == k else 0.0
                        max_err = max(max_err, abs(T_check[i, j, k, l] - target))
        
        if max_err < 1e-8:
            return {
                "U": U_half.tolist(),
                "V": V_half.tolist(),
                "W": W_half.tolist()
            }
            
        return None

    for R in range(8, 12):
        result = try_rank(R)
        if result:
            return result
            
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
