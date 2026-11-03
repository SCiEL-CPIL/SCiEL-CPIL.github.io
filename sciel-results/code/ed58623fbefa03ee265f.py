"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 105
    
    def verify(U, V, W, tol=1e-6):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_decomposition(r, seed=42, max_iters=3000):
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(seed)
        
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        def compute_error_and_grad(U, V, W):
            err = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for (i, j, k) in indices:
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                err_val = s - 1.0
                err += err_val ** 2
                prod = V[v_idx, :] * W[w_idx, :]
                dU[u_idx, :] += 2 * err_val * prod
                dV[v_idx, :] += 2 * err_val * U[u_idx, :] * W[w_idx, :]
                dW[w_idx, :] += 2 * err_val * U[u_idx, :] * V[v_idx, :]
            return err, dU, dV, dW
        
        lr = 0.05
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            err, dU, dV, dW = compute_error_and_grad(U, V, W)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if err < 1e-8:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r, tol=1e-8):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
            
            if it % 5 == 0:
                lr *= 0.99
        
        return None
    
    for r in range(16, 37):
        if time.time() > deadline - 25:
            break
        for seed in range(15):
            if time.time() > deadline - 20:
                break
            result = try_decomposition(r, seed=seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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

