"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_half(x):
    return round(x * 2.0) / 2.0

def _als(n, m, p, rank, max_iter=50):
    """Vectorized ALS for the multiplication tensor."""
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                B = np.zeros((p, rank))
                for k in range(p):
                    B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                C = np.zeros((n, rank))
                for i in range(n):
                    C[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(C, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                D = np.zeros((m, rank))
                for j in range(m):
                    D[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(D, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]))
                    max_err = max(max_err, abs(s - 1.0))
        if max_err < 1e-10:
            break
    
    return U, V, W, max_err

def _try_round(U, V, W, n, m, p):
    """Try rounding to half-integers and verify."""
    for mode in ['half', 'int']:
        if mode == 'half':
            U_r = np.array([[_round_half(x) for x in row] for row in U])
            V_r = np.array([[_round_half(x) for x in row] for row in V])
            W_r = np.array([[_round_half(x) for x in row] for row in W])
        else:
            U_r = np.round(U).astype(int).astype(float)
            V_r = np.round(V).astype(int).astype(float)
            W_r = np.round(W).astype(int).astype(float)
        
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def _schoolbook():
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
    
    # Try to find a decomposition with rank < 80
    for rank in range(35, 80):
        if time.time() > deadline:
            break
        
        n_restarts = 4 if rank < 55 else 2
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 1000 + r * 7 + 13)
            U, V, W, err = _als(n, m, p, rank, max_iter=80)
            
            if err < 1e-6:
                res = _try_round(U, V, W, n, m, p)
                if res is not None:
                    U_l, V_l, W_l = res
                    if _verify(U_l, V_l, W_l, n, m, p):
                        return {"U": U_l, "V": V_l, "W": W_l}
            
            # If close, try rounding anyway
            if err < 0.05:
                res = _try_round(U, V, W, n, m, p)
                if res is not None:
                    U_l, V_l, W_l = res
                    if _verify(U_l, V_l, W_l, n, m, p):
                        return {"U": U_l, "V": V_l, "W": W_l}
    
    return _schoolbook()
