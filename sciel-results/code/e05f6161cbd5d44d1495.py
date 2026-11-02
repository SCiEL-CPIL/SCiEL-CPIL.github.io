import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Schoolbook decomposition
    rank_fb = n * m * p  # 48
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-6):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j, :], np.multiply(V_arr[j*p+k, :], W_arr[k*n+i, :]))
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def try_find_decomp(r, num_restarts=3, max_iters=3000):
        if time.time() > deadline - 30:
            return None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(2000 + restart)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            lr = 0.05
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline - 30:
                    break
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                loss = 0.0
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            uvw = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                            val = np.sum(uvw)
                            err = val - 1.0
                            loss += err * err
                            factor = 2.0 * err
                            grad_U[i*m+j, :] += factor * V[j*p+k, :] * W[k*n+i, :]
                            grad_V[j*p+k, :] += factor * U[i*m+j, :] * W[k*n+i, :]
                            grad_W[k*n+i, :] += factor * U[i*m+j, :] * V[j*p+k, :]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-10:
                    break
                
                if it % 500 == 0 and it > 0:
                    lr *= 0.9
            
            if best_loss < 1e-4:
                # Try snapping to half-integers
                for snap in [2, 4, 8, 16]:
                    Us = np.round(best_U * snap) / snap
                    Vs = np.round(best_V * snap) / snap
                    Ws = np.round(best_W * snap) / snap
                    if verify(Us, Vs, Ws):
                        return Us, Vs, Ws
                
                if verify(best_U, best_V, best_W, tol=1e-4):
                    return best_U, best_V, best_W
        
        return None
    
    # Try to find lower rank
    min_rank = max(n*m, m*p, p*n)  # 16
    
    for r in range(min_rank, rank_fb + 1):
        if time.time() > deadline - 40:
            break
        
        result = try_find_decomp(r)
        if result is not None:
            U_res, V_res, W_res = result
            U_list = [[float(x) for x in row] for row in U_res]
            V_list = [[float(x) for x in row] for row in V_res]
            W_list = [[float(x) for x in row] for row in W_res]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
