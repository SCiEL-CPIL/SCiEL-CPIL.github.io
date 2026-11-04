"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 150.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using numpy einsum."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Check half-integer constraint
    for arr in [Ua, Va, Wa]:
        x2 = 2.0 * arr
        if not np.allclose(x2, np.round(x2), atol=1e-9):
            return False
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _snap_half(x):
    return round(2.0 * x) / 2.0

def _snap_array(arr):
    return np.vectorize(_snap_half)(arr)

def _als_decompose(n, m, p, rank, max_iter=5000, tol=1e-12, seed=0):
    """Alternating least squares to find decomposition."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 3.0:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k] * W[k*n+i]
                try:
                    U[i*m+j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j] * W[k*n+i]
                try:
                    V[j*p+k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j] * V[j*p+k]
                try:
                    W[k*n+i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.max(np.abs(T - 1.0))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    # Snap to half-integers
    U_s = _snap_array(best_U)
    V_s = _snap_array(best_V)
    W_s = _snap_array(best_W)
    
    if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p):
        return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    # Try without snapping
    if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    min_rank = max(n*m, m*p, n*p)
    
    for rank in range(min_rank, best_rank + 1):
        if _remaining() < 15.0:
            break
        
        found = False
        for seed in range(20):
            if _remaining() < 5.0:
                break
            result = _als_decompose(n, m, p, rank, max_iter=2000, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
    
    return best
