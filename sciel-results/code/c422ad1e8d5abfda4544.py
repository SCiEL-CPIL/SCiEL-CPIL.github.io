"""Matrix multiplication 3x4x8 rank-minimal search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
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
    
    def verify(U, V, W, tol=1e-10):
        r = len(U[0])
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    best = schoolbook()
    
    def try_rank(r, max_attempts=100):
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            random.seed(2000 + attempt)
            U = np.random.randint(-2, 3, size=(n*m, r)).astype(np.float64)
            V = np.random.randint(-2, 3, size=(m*p, r)).astype(np.float64)
            W = np.random.randint(-2, 3, size=(p*n, r)).astype(np.float64)
            
            for it in range(200):
                if time.time() > deadline:
                    return None
                
                # Fix U,V; solve for W
                M = np.zeros((n, p, r))
                for i in range(n):
                    for k in range(p):
                        for c in range(r):
                            M[i,k,c] = sum(U[i*m+j, c] * V[j*p+k, c] for j in range(m))
                
                W_new = np.zeros((p*n, r))
                ok = True
                for i in range(n):
                    for k in range(p):
                        mvec = M[i,k,:]
                        norm_sq = np.dot(mvec, mvec)
                        if norm_sq < 1e-15:
                            ok = False
                            break
                        W_new[k*n+i,:] = mvec / norm_sq
                if not ok:
                    break
                W = W_new
                
                # Fix U,W; solve for V
                N = np.zeros((m, p, r))
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            N[j,k,c] = sum(U[i*m+j, c] * W[k*n+i, c] for i in range(n))
                
                V_new = np.zeros((m*p, r))
                ok = True
                for j in range(m):
                    for k in range(p):
                        nvec = N[j,k,:]
                        norm_sq = np.dot(nvec, nvec)
                        if norm_sq < 1e-15:
                            ok = False
                            break
                        V_new[j*p+k,:] = nvec / norm_sq
                if not ok:
                    break
                V = V_new
                
                # Fix V,W; solve for U
                P = np.zeros((n, m, r))
                for i in range(n):
                    for j in range(m):
                        for c in range(r):
                            P[i,j,c] = sum(V[j*p+k, c] * W[k*n+i, c] for k in range(p))
                
                U_new = np.zeros((n*m, r))
                ok = True
                for i in range(n):
                    for j in range(m):
                        pvec = P[i,j,:]
                        norm_sq = np.dot(pvec, pvec)
                        if norm_sq < 1e-15:
                            ok = False
                            break
                        U_new[i*m+j,:] = pvec / norm_sq
                if not ok:
                    break
                U = U_new
                
                # Check and try to round
                if verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-8):
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), tol=1e-10):
                        def to_list(arr):
                            return [[int(x) if x == int(x) else float(x) for x in row] for row in arr]
                        return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
        
        return None
    
    for r in range(95, 79, -1):
        if time.time() > deadline:
            break
        result = try_rank(r, max_attempts=50)
        if result is not None:
            return result
    
    return best
# EVOLVE-BLOCK-END
