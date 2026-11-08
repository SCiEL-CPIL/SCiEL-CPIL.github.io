"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def make_schoolbook():
        r = n*m*p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def residuals(U, V, W):
        r = U.shape[1]
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[idx] = 1.0 - np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    idx += 1
        return res
    
    def try_rank(r, max_iters=3000, seed=42):
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(nm, r) * 0.3
        V = np.random.randn(mp, r) * 0.3
        W = np.random.randn(pn, r) * 0.3
        
        lr = 0.003
        best_res = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            res = residuals(U, V, W)
            max_res = np.max(np.abs(res))
            
            if max_res < best_res:
                best_res = max_res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if max_res < 1e-10:
                return U, V, W
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ri = res[i*m*p + j*p + k]
                        dU[i*m+j, :] += 2*ri * V[j*p+k, :] * W[k*n+i, :]
                        dV[j*p+k, :] += 2*ri * U[i*m+j, :] * W[k*n+i, :]
                        dW[k*n+i, :] += 2*ri * U[i*m+j, :] * V[j*p+k, :]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if it % 100 == 99:
                lr *= 0.95
        
        if best_res < 1e-8:
            return best_U, best_V, best_W
        return None
    
    # Try to find rank < 32
    for r in range(20, 32):
        if time.time() > deadline - 15:
            break
        for seed in [r*7, r*13, r*17]:
            if time.time() > deadline - 5:
                break
            result = try_rank(r, max_iters=2000, seed=seed)
            if result:
                U, V, W = result
                U2, V2, W2 = np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2
                if verify(U2, V2, W2, 1e-9):
                    U, V, W = U2, V2, W2
                U_l = [[float(U[i,j]) for j in range(r)] for i in range(nm)]
                V_l = [[float(V[i,j]) for j in range(r)] for i in range(mp)]
                W_l = [[float(W[i,j]) for j in range(r)] for i in range(pn)]
                return {"U": U_l, "V": V_l, "W": W_l}
    
    # Fallback to schoolbook rank 32
    U, V, W = make_schoolbook()
    r = n*m*p
    U_l = [[float(U[i,j]) for j in range(r)] for i in range(nm)]
    V_l = [[float(V[i,j]) for j in range(r)] for i in range(mp)]
    W_l = [[float(W[i,j]) for j in range(r)] for i in range(pn)]
    return {"U": U_l, "V": V_l, "W": W_l}
# EVOLVE-BLOCK-END
