import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook as fallback
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, rank):
        """Verify exact decomposition with half-integer entries"""
        # Check all entries are half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        # Check tensor equation
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def try_rank(r, max_iters=2000, seed=42):
        """Try to find rank-r decomposition using ALS"""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    # U[i*m+j, :] * (V[j*p+k,:] * W[k*n+i,:] for all k) = 1
                    # This is: U[i*m+j,:] @ diag(V[j*p+k,:]*W[k*n+i,:]) ... 
                    # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    # Let M_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product)
                    # Then: sum_c U[i*m+j,c] * M_k[c] = 1 for all k
                    # So: U[i*m+j,:] @ M^T = ones(p)
                    # Where M is p x r with M[k,c] = V[j*p+k,c]*W[k*n+i,c]
                    M = np.outer(W[:p*n].reshape(p, n, n).transpose(1,0,2).reshape(p, n*n)[:, :p*n] if False else W, 1) # This is getting complicated
                    # Let me just do it directly
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve: U[i*m+j,:] @ M.T = ones(p)
                    # U[i*m+j,:] = ones(p) @ pinv(M.T)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                    # M_i = U[i*m+j,:] * W[k*n+i,:]
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve
            for k in range(p):
                for i in range(n):
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if it % 10 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                            err += (s - 1.0)**2
                if err < 1e-12:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    # Verify
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(r):
                                    s += U_r[i*m+j,c] * V_r[j*p+k,c] * W_r[k*n+i,c]
                                if abs(s - 1.0) > 1e-9:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        return (U_r.tolist(), V_r.tolist(), W_r.tolist(), r)
        
        return None
    
    # Try decreasing ranks
    for r in range(best_rank - 1, 55, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W, rank = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
    
    return best_result
