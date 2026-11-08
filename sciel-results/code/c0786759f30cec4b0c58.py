import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 125  # seconds
    
    def check_valid(U, V, W, tol=1e-8):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = V[j*p+k, :] * W[k*n+i, :]
                    val = np.dot(U[i*m+j, :], prod)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    def alternate(U, V, W, max_iter=5000, tol=1e-14):
        rank = U.shape[1]
        for it in range(max_iter):
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    U_new[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    V_new[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    W_new[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            if diff < tol:
                break
        return U, V, W
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    for rank in range(30, best_rank):
        if time.time() - start_time > time_limit:
            break
        found = False
        for trial in range(300):
            if time.time() - start_time > time_limit:
                break
            np.random.seed(1000 + rank * 100 + trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            U, V, W = alternate(U, V, W, max_iter=3000, tol=1e-14)
            if check_valid(U, V, W, tol=1e-8):
                found = True
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        if not found:
            continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
