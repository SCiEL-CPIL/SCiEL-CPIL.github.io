import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    timeout = 120.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    def compute_loss(U, V, W):
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    total += (val - 1.0) ** 2
        return total
    
    def solve_for_W(U, V, r):
        W = np.zeros((p * n, r))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W_col, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = W_col
                except:
                    W[k*n+i, :] = 0
        return W
    
    def solve_for_U(U, V, W, r):
        U_new = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U_col, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = U_col
                except:
                    U_new[i*m+j, :] = 0
        return U_new
    
    def solve_for_V(U, V, W, r):
        V_new = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V_col, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = V_col
                except:
                    V_new[j*p+k, :] = 0
        return V_new
    
    def try_rank(r, max_sweeps=30, n_seeds=3):
        for seed in range(n_seeds):
            np.random.seed(42 + seed)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for sweep in range(max_sweeps):
                W = solve_for_W(U, V, r)
                U = solve_for_U(U, V, W, r)
                V = solve_for_V(U, V, W, r)
                
                loss = compute_loss(U, V, W)
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-10:
                    break
                
                if time.time() - start_time > timeout - 10:
                    return None
            
            if best_loss < 1e-8:
                # Polish with tighter tolerance
                U_polish = best_U.copy()
                V_polish = best_V.copy()
                W_polish = best_W.copy()
                for _ in range(10):
                    W_polish = solve_for_W(U_polish, V_polish, r)
                    U_polish = solve_for_U(U_polish, V_polish, W_polish, r)
                    V_polish = solve_for_V(U_polish, V_polish, W_polish, r)
                
                U_r = np.round(U_polish * 2) / 2
                V_r = np.round(V_polish * 2) / 2
                W_r = np.round(W_polish * 2) / 2
                
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    return {
                        "U": [[float(x) for x in row] for row in U_r],
                        "V": [[float(x) for x in row] for row in V_r],
                        "W": [[float(x) for x in row] for row in W_r]
                    }
        
        return None
    
    # Try ranks from 71 down to 50, with time budget
    for r in range(71, 49, -1):
        if time.time() - start_time > timeout - 15:
            break
        
        result = try_rank(r, max_sweeps=25, n_seeds=2)
        if result is not None:
            return result
    
    return schoolbook()
