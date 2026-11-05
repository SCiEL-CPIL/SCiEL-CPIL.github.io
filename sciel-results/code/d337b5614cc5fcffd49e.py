import time
import numpy as np

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1.0) > 1e-7:
                    return False
    return True

def _als_refine(U, V, W, n, m, p, max_iters=20000, tol=1e-9):
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    rank = U.shape[1]
    
    for it in range(max_iters):
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                try:
                    w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = w
                except:
                    pass
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                try:
                    u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = u
                except:
                    pass
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                try:
                    v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p+k, :] = v
                except:
                    pass
        
        if it % 500 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0)**2
            if np.sqrt(err) < tol:
                break
    
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err += (val - 1.0)**2
    return np.sqrt(err) < tol, U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return U, V, W

def construct():
    n, m, p = 3, 3, 6
    t0 = time.time()
    deadline = t0 + 135
    
    # Try to find rank-46 by solving 3x3x3 rank-23
    # Use multiple random restarts for 3x3x3 rank 23
    n3, m3, p3 = 3, 3, 3
    rank23 = 23
    
    for seed in range(20):
        if time.time() > deadline - 15:
            break
        np.random.seed(seed * 137 + 42)
        U3 = np.random.randn(n3*m3, rank23) * 0.3
        V3 = np.random.randn(m3*p3, rank23) * 0.3
        W3 = np.random.randn(p3*n3, rank23) * 0.3
        
        ok, U3, V3, W3 = _als_refine(U3, V3, W3, n3, m3, p3, max_iters=15000, tol=1e-9)
        if ok:
            # Build 3x3x6 rank 46
            rank46 = 46
            U46 = [[0.0]*rank46 for _ in range(n*m)]
            V46 = [[0.0]*rank46 for _ in range(m*p)]
            W46 = [[0.0]*rank46 for _ in range(p*n)]
            
            for c in range(rank23):
                for i in range(n):
                    for j in range(m):
                        U46[i*m+j][c] = float(U3[i*m3+j, c])
                        U46[i*m+j][rank23+c] = float(U3[i*m3+j, c])
                for j in range(m):
                    for k in range(3):
                        V46[j*p+k][c] = float(V3[j*m3+k, c])
                        V46[j*p+3+k][rank23+c] = float(V3[j*m3+k, c])
                for k in range(3):
                    for i in range(n):
                        W46[k*n+i][c] = float(W3[k*n3+i, c])
                        W46[(k+3)*n+i][rank23+c] = float(W3[k*n3+i, c])
            
            if _validate(U46, V46, W46, n, m, p):
                return {"U": U46, "V": V46, "W": W46}
    
    # Fallback
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
