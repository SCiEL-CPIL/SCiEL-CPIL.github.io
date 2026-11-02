import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    triplets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    def get_residuals(U, V, W):
        res = {}
        for (i, j, k) in triplets:
            val = np.sum(U[u_idx(i, j)] * V[v_idx(j, k)] * W[w_idx(k, i)])
            res[(i, j, k)] = val - 1.0
        return res

    def total_error(U, V, W):
        res = get_residuals(U, V, W)
        return float(np.sum([r*r for r in res.values()]))

    def snap_to_half_int(X):
        return np.round(X * 2) / 2.0

    def optimize(U, V, W, max_iters=300, tol=1e-10, deadline=None):
        lr = 0.05
        best_err = total_error(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
                
            res = get_residuals(U, V, W)
            err = total_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for (i, j, k), r in res.items():
                ui = u_idx(i, j)
                vi = v_idx(j, k)
                wi = w_idx(k, i)
                
                dU[ui] += 2 * r * V[vi] * W[wi]
                dV[vi] += 2 * r * U[ui] * W[wi]
                dW[wi] += 2 * r * U[ui] * V[vi]
                
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            lr *= 0.99
            
        return best_U, best_V, best_W

    start_time = time.time()
    deadline = start_time + 110
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    for r in range(16, 25):
        if time.time() > deadline:
            break
            
        num_restarts = 100 if r == 16 else 50
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            scale = 1.0 / np.sqrt(r)
            U_init = scale * np.random.randn(n*m, r)
            V_init = scale * np.random.randn(m*p, r)
            W_init = scale * np.random.randn(p*n, r)
            
            U_opt, V_opt, W_opt = optimize(U_init, V_init, W_init, max_iters=200, tol=1e-10, deadline=deadline)
            
            err = total_error(U_opt, V_opt, W_opt)
            
            if err < 1e-4:
                U_snap = snap_to_half_int(U_opt)
                V_snap = snap_to_half_int(V_opt)
                W_snap = snap_to_half_int(W_opt)
                
                err_snap = total_error(U_snap, V_snap, W_snap)
                
                if err_snap < 1e-9:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                
            if err < best_err:
                best_err = err
                best_U = U_opt
                best_V = V_opt
                best_W = W_opt
                best_rank = r

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
