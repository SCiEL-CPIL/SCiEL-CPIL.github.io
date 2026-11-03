"""Combined program: Block decomposition for 3x3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook decomposition as fallback
    def schoolbook(n, m, p):
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
        return U, V, W, rank
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is exact"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        rank = U.shape[1]
        T_recon = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            u_col = U[:, c].reshape(n, m)
            v_col = V[:, c].reshape(m, p)
            w_col = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        
        T_target = np.ones((n, m, p), dtype=float)
        return np.max(np.abs(T_recon - T_target)) < tol
    
    def try_optimization(r, n, m, p, max_iters=300, tol=1e-9, seed=42, time_limit=20):
        """Try to find a rank-r decomposition using optimization"""
        start = time.time()
        np.random.seed(seed)
        
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            T_curr = np.zeros((n, m, p), dtype=float)
            for c in range(r):
                u_col = U[:, c].reshape(n, m)
                v_col = V[:, c].reshape(m, p)
                w_col = W[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_curr[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
            
            T_target = np.ones((n, m, p), dtype=float)
            err = np.max(np.abs(T_curr - T_target))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
            
            lr = 0.01
            for c in range(r):
                u_col = U[:, c].reshape(n, m)
                v_col = V[:, c].reshape(m, p)
                w_col = W[:, c].reshape(p, n)
                
                grad_U = np.zeros((n, m), dtype=float)
                grad_V = np.zeros((m, p), dtype=float)
                grad_W = np.zeros((p, n), dtype=float)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual = T_curr[i, j, k] - 1.0
                            grad_U[i, j] += residual * v_col[j, k] * w_col[k, i]
                            grad_V[j, k] += residual * u_col[i, j] * w_col[k, i]
                            grad_W[k, i] += residual * u_col[i, j] * v_col[j, k]
                
                U[:, c] -= lr * grad_U.flatten()
                V[:, c] -= lr * grad_V.flatten()
                W[:, c] -= lr * grad_W.flatten()
        
        if best_err < tol:
            return best_U.tolist(), best_V.tolist(), best_W.tolist(), r
        return None
    
    U_school, V_school, W_school, rank_school = schoolbook(n, m, p)
    best_result = (U_school, V_school, W_school, rank_school)
    
    for r in range(21, rank_school):
        if time.time() > deadline - 60:
            break
        
        result = try_optimization(r, n, m, p, max_iters=200, tol=1e-9, seed=r, time_limit=15)
        if result is not None:
            U_try, V_try, W_try, r_try = result
            if verify_decomposition(U_try, V_try, W_try, n, m, p, tol=1e-9):
                best_result = (U_try, V_try, W_try, r_try)
                break
    
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
