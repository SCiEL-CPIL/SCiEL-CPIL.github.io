import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    def verify_fast(U_arr, V_arr, W_arr, n, m, p, tol=1e-6):
        r = U_arr.shape[1]
        # Diagonal check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k]) @ W_arr[k*n+i]
                    if abs(s - 1.0) > tol:
                        return False
        # Off-diagonal check
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j == j2: continue
                    for k in range(p):
                        s = np.dot(U_arr[i*m+j], V_arr[j2*p+k]) @ W_arr[k*n+i]
                        if abs(s) > tol:
                            return False
        return True

    def verify(U, V, W, n, m, p, tol=1e-6):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        return verify_fast(U_arr, V_arr, W_arr, n, m, p, tol)

    def schoolbook(n, m, p):
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
        return U, V, W

    def error_func_factory(rank):
        def error_func(x):
            U = x[:n*m*rank].reshape(n*m, rank)
            V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            err = 0.0
            # Diagonal
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k]) @ W[k*n+i]
                        err += (s - 1.0)**2
            # Off-diagonal
            for i in range(n):
                for j in range(m):
                    for j2 in range(m):
                        if j == j2: continue
                        for k in range(p):
                            s = np.dot(U[i*m+j], V[j2*p+k]) @ W[k*n+i]
                            err += s**2
            return err
        return error_func

    def try_rank(rank, n_seeds=5):
        if time.time() > deadline - 20:
            return None
        
        for seed in range(n_seeds):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed * 1000 + rank)
            
            # Initialize with small random values
            scale = 1.0
            U_init = np.random.randn(n*m, rank) * scale
            V_init = np.random.randn(m*p, rank) * scale
            W_init = np.random.randn(p*n, rank) * scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            error_func = error_func_factory(rank)
            
            try:
                res = minimize(error_func, x0, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.fun < 1e-6:
                    U_sol = res.x[:n*m*rank].reshape(n*m, rank)
                    V_sol = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_sol = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Try rounding to half-integers
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    if verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p):
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                    
                    # Try rounding to integers
                    U_int = np.round(U_sol)
                    V_int = np.round(V_sol)
                    W_int = np.round(W_sol)
                    if verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
            except Exception:
                pass
        
        return None

    # Try ranks from 49 down to 30
    for rank in range(49, 29, -1):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(rank, n_seeds=3)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
