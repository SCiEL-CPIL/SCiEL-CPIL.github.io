"""Matrix multiplication 3x5x7 - tensor rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
_START = time.time()
_DEADLINE = 145.0

def _left():
    return _DEADLINE - (time.time() - _START)

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def schoolbook():
        r = n*m*p
        U = [[0]*r for _ in range(nm)]
        V = [[0]*r for _ in range(mp)]
        W = [[0]*r for _ in range(pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, r, tol=1e-8):
        Ua = np.array(U, dtype=np.float64).reshape(n, m, r)
        Va = np.array(V, dtype=np.float64).reshape(m, p, r)
        Wa = np.array(W, dtype=np.float64).reshape(p, n, r)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += Ua[i,j,c]*Va[j,k,c]*Wa[k,i,c]
                    if abs(s-1.0) > tol:
                        return False
        return True
    
    # Search for lower rank using random initialization + least-squares on linearized system
    # For rank r, fix U and V, then W is determined linearly? No, it's trilinear.
    
    # Use alternating minimization:
    # Fix V, W: solve for U linearly (each U[i*m+j,c] appears linearly)
    # Actually no - it's product of three, so fixing two makes it linear in the third.
    
    # Alternating:
    # 1. Fix V, W: for each (i,j), U[i*m+j,:] = solve min sum_k (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
    #    This is: for each (i,j), minimize ||b - M @ u||^2 where u = U[i*m+j,:], 
    #    M[c, k] = V[j*p+k, c] * W[k*n+i, c], b[k] = 1
    #    Wait, that's not right either. Let me re-derive.
    #    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    #    Fix V, W. For fixed (i,j): sum_c U[i,j,c] * (V[j,k,c]*W[k,i,c]) = 1 for all k
    #    Let A[i,j] be a p x r matrix with A[i,j][k,c] = V[j*p+k,c]*W[k*n+i,c]
    #    Then A @ U[i,j,:] = 1_p (vector of ones)
    #    Solve least squares.
    
    def alt_minimize(U, V, W, r, iters=100):
        Ua = np.array(U, dtype=np.float64).reshape(n, m, r)
        Va = np.array(V, dtype=np.float64).reshape(m, p, r)
        Wa = np.array(W, dtype=np.float64).reshape(p, n, r)
        
        for it in range(iters):
            if _left() < 2:
                break
            # Update U: for each (i,j), solve A @ u = 1
            for i in range(n):
                for j in range(m):
                    # A[k, c] = Va[j,k,c] * Wa[k,i,c]
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = Va[j, k, :] * Wa[k, i, :]
                    # Solve A @ u = ones(p)
                    try:
                        u, *_ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        Ua[i, j, :] = u
                    except:
                        pass
            # Update V: for each (j,k), solve A @ v = 1
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = Ua[i, j, :] * Wa[k, i, :]
                    try:
                        v, *_ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        Va[j, k, :] = v
                    except:
                        pass
            # Update W: for each (k,i), solve A @ w = 1
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = Ua[i, j, :] * Va[j, k, :]
                    try:
                        w, *_ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        Wa[k, i, :] = w
                    except:
                        pass
        
        return Ua.flatten().tolist(), Va.flatten().tolist(), Wa.flatten().tolist()
    
    # Try various ranks
    best = schoolbook()
    for r in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        if _left() < 5:
            break
        rng = np.random.RandomState(r)
        U0 = rng.randn(n*m, r)
        V0 = rng.randn(m*p, r)
        W0 = rng.randn(p*n, r)
        budget_iters = min(200, int(_left() * 50))
        Uf, Vf, Wf = alt_minimize(U0, V0, W0, r, iters=budget_iters)
        if verify(Uf, Vf, Wf, r):
            best = {"U": Uf, "V": Vf, "W": Wf}
            break
    
    return best
