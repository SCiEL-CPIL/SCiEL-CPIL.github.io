import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(result, T, atol=tol)

def _try_rank(r, T, max_iter=200, tol=1e-10, seed=0, time_limit=5.0):
    start = time.time()
    np.random.seed(seed)
    nm, mp, pn = T.shape
    U = np.random.randn(nm, r)
    V = np.random.randn(mp, r)
    W = np.random.randn(pn, r)
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Solve for U
        # T[a] (mp*pn) = U[a] @ M_U.T where M_U[:,rr] = V[:,rr]⊗W[:,rr]
        M_U = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, r)
        pinv_M_U = np.linalg.pinv(M_U.T)
        U_new = T.reshape(nm, mp*pn) @ pinv_M_U
        
        # Solve for V
        M_V = np.einsum('ar,cr->acr', U_new, W).reshape(nm*pn, r)
        pinv_M_V = np.linalg.pinv(M_V.T)
        V_new = T.reshape(nm*pn, mp).T @ pinv_M_V
        
        # Solve for W
        M_W = np.einsum('ar,br->abr', U_new, V_new).reshape(nm*mp, r)
        pinv_M_W = np.linalg.pinv(M_W.T)
        W_new = T.reshape(nm*mp, pn).T @ pinv_M_W
        
        if np.allclose(U_new, U, atol=1e-12) and np.allclose(V_new, V, atol=1e-12) and np.allclose(W_new, W, atol=1e-12):
            U, V, W = U_new, V_new, W_new
            break
        
        U, V, W = U_new, V_new, W_new
    
    if _verify(U, V, W, T, tol=tol):
        return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    T = _build_tensor()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    lower_bound = max(nm, mp, pn)
    
    for r in range(lower_bound, 37):
        if time.time() > deadline:
            break
        for seed in range(50):
            if time.time() > deadline:
                break
            result = _try_rank(r, T, max_iter=100, tol=1e-10, seed=seed, time_limit=2.0)
            if result is not None:
                U, V, W = result
                return {"U": [[float(x) for x in row] for row in U],
                        "V": [[float(x) for x in row] for row in V],
                        "W": [[float(x) for x in row] for row in W]}
    
    # Fallback
    rank = n*m*p
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
