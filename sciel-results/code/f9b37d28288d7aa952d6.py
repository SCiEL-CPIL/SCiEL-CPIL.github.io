"""Rewritten program: numerical search for low-rank decomposition of 2x3x6 matmul tensor."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    def verify(U, V, W, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_iters=3000, seed=42):
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((r, p))
                    for k in range(p):
                        M[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    N = np.zeros((r, n))
                    for i in range(n):
                        N[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    P = np.zeros((r, m))
                    for j in range(m):
                        P[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            if it % 150 == 0 and verify(U, V, W, tol=1e-8):
                return (U.copy(), V.copy(), W.copy())
        
        if verify(U, V, W, tol=1e-8):
            return (U, V, W)
        return None
    
    # Try ranks from low to high with multiple seeds
    best_result = None
    for r in range(16, 37):
        if time.time() > deadline - 20:
            break
        for seed in [r, r+100, r+200, r+300, r+400, r+500, r+600, r+700]:
            if time.time() > deadline - 15:
                break
            result = try_rank(r, max_iters=2000, seed=seed)
            if result is not None:
                best_result = (r, result)
                break
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback: schoolbook
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
    
    r, (U, V, W) = best_result
    U_list = [[round(float(U[i,j]), 10) for j in range(r)] for i in range(n*m)]
    V_list = [[round(float(V[i,j]), 10) for j in range(r)] for i in range(m*p)]
    W_list = [[round(float(W[i,j]), 10) for j in range(r)] for i in range(p*n)]
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                if abs(s - 1.0) > 1e-6:
                    rank = n * m * p
                    U_fb = [[0] * rank for _ in range(n * m)]
                    V_fb = [[0] * rank for _ in range(m * p)]
                    W_fb = [[0] * rank for _ in range(p * n)]
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][column] = 1
                                V_fb[jj * p + kk][column] = 1
                                W_fb[kk * n + ii][column] = 1
                                column += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
