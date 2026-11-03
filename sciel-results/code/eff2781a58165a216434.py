"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    
    # Build the target tensor T of shape (dim_W, dim_U, dim_V)
    # T[a, b, c] = 1 if indices match multiplication term
    # a = k*n + i, b = i*m + j, c = j*p + k
    T = np.zeros((dim_W, dim_U, dim_V))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                T[a, b, c] = 1.0

    T_flat = T.flatten()

    def check_solution(U, V, W):
        """Check if U, V, W exactly decompose T."""
        T_rec = np.zeros((dim_W, dim_U, dim_V))
        for rr in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', W[:, rr], U[:, rr], V[:, rr])
        return np.allclose(T_rec, T, atol=1e-10)

    def try_round_and_check(U, V, W, scale=2):
        """Try to round U, V, W to nearest 1/scale and check."""
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        
        T_rec = np.zeros((dim_W, dim_U, dim_V))
        for rr in range(U_r.shape[1]):
            T_rec += np.einsum('a,b,c->abc', W_r[:, rr], U_r[:, rr], V_r[:, rr])
        
        if np.allclose(T_rec, T, atol=1e-10):
            return U_r, V_r, W_r
        return None

    def objective(params, r):
        """Objective function for optimization: sum of squared errors."""
        U = params[0:dim_U * r].reshape(dim_U, r)
        V = params[dim_U * r:dim_U * r + dim_V * r].reshape(dim_V, r)
        W = params[dim_U * r + dim_V * r:].reshape(dim_W, r)
        
        T_rec = np.zeros((dim_W, dim_U, dim_V))
        for rr in range(r):
            T_rec += np.einsum('a,b,c->abc', W[:, rr], U[:, rr], V[:, rr])
        
        error = np.sum((T - T_rec) ** 2)
        return error

    best_result = None
    best_rank = 24
    
    # Theoretical min rank for 2x3x4 is 19. Schoolbook is 24.
    # We try 19 first, then 20, 21, 22, 23.
    
    max_restarts = 50
    max_iters = 1000
    
    for r in range(19, 24):
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize parameters
            # Use small random values
            U = np.random.randn(dim_U, r) * 0.1
            V = np.random.randn(dim_V, r) * 0.1
            W = np.random.randn(dim_W, r) * 0.1
            
            # Flatten
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Optimize
            res = minimize(objective, params0, args=(r,), method='Nelder-Mead', 
                           options={'maxiter': max_iters, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < 1e-6:
                # Extract optimized U, V, W
                U_opt = res.x[0:dim_U * r].reshape(dim_U, r)
                V_opt = res.x[dim_U * r:dim_U * r + dim_V * r].reshape(dim_V, r)
                W_opt = res.x[dim_U * r + dim_V * r:].reshape(dim_W, r)
                
                # Try rounding
                rounded = try_round_and_check(U_opt, V_opt, W_opt, scale=2)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_result = (U_r, V_r, W_r, r)
                    found = True
                    break
                
                # If rounding to 0.5 failed, try 1.0 (integer)
                rounded_int = try_round_and_check(U_opt, V_opt, W_opt, scale=1)
                if rounded_int is not None:
                    U_r, V_r, W_r = rounded_int
                    best_result = (U_r, V_r, W_r, r)
                    found = True
                    break

        if found:
            break

    if best_result is not None:
        U_r, V_r, W_r, r = best_result
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Fallback: Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
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
