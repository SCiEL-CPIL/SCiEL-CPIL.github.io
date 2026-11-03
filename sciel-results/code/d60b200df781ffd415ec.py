import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
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
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
                
    T_flat = T.flatten()
    
    def residual(params, R):
        U = params[:R * n * m].reshape(R, n * m)
        V = params[R * n * m : R * n * m + R * m * p].reshape(R, m * p)
        W = params[R * n * m + R * m * p : R * n * m + R * m * p + R * p * n].reshape(R, p * n)
        
        T_comp = np.zeros((n, m, p))
        for r in range(R):
            u = U[r].reshape(n, m)
            v = V[r].reshape(m, p)
            w = W[r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_comp[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                        
        return (T_comp - T).flatten()
        
    start_time = time.time()
    deadline = 120
    best_result = None
    best_rank = rank_sb
    
    for R in range(55, 22, -1):
        if time.time() > start_time + deadline - 10:
            break
        for attempt in range(5):
            if time.time() > start_time + deadline - 10:
                break
            np.random.seed(attempt * 100 + R)
            params0 = np.random.randn(R * (n * m + m * p + p * n))
            try:
                res = least_squares(residual, params0, args=(R,), method='lm')
                if res.success and np.max(np.abs(res.fun)) < 1e-6:
                    U = res.x[:R * n * m].reshape(R, n * m)
                    V = res.x[R * n * m : R * n * m + R * m * p].reshape(R, m * p)
                    W = res.x[R * n * m + R * m * p : R * n * m + R * m * p + R * p * n].reshape(R, p * n)
                    
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    
                    T_comp = np.zeros((n, m, p))
                    for r in range(R):
                        u = U_round[r].reshape(n, m)
                        v = V_round[r].reshape(m, p)
                        w = W_round[r].reshape(p, n)
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T_comp[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                                    
                    if np.max(np.abs(T_comp - T)) < 1e-9:
                        best_result = (U_round, V_round, W_round)
                        best_rank = R
                        break
            except Exception:
                pass
        if best_result:
            break
            
    if best_result:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
