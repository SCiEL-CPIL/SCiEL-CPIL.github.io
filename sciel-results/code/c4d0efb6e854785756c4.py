import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    T = np.ones((n, m, p))
    R_sb = n * m * p  # 84
    
    # Schoolbook decomposition
    U_sb = np.zeros((n * m, R_sb))
    V_sb = np.zeros((m * p, R_sb))
    W_sb = np.zeros((p * n, R_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def check(U, V, W, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def als(r, iters=50, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        for _ in range(iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B = (V[j*p:(j+1)*p] * W.reshape(p, n, -1)[:, i, :]).T
                    U[i*m+j] = np.linalg.lstsq(B, T[i,j], rcond=None)[0]
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = U.reshape(n, m, -1)[:, j, :] * W[k*n:(k+1)*n].T
                    V[j*p+k] = np.linalg.lstsq(B, T[:,j,k], rcond=None)[0]
            
            # Update W
            for k in range(p):
                for i in range(n):
                    B = U[i*m:(i+1)*m] * V.reshape(m, p, -1)[:, k, :].T
                    W[k*n+i] = np.linalg.lstsq(B, T[:,:,k], rcond=None)[0]
            
            if check(U, V, W):
                return (U, V, W)
        return None
    
    # Try ranks from 83 down to 28
    for r in range(R_sb - 1, 27, -1):
        if time.time() > deadline - 20:
            break
        for seed in range(2):
            if time.time() > deadline - 20:
                break
            res = als(r, iters=30, seed=seed)
            if res:
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

