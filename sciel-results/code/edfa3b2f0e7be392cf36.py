import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 110

def _elapsed():
    return time.time() - START_TIME

def _verify(U, V, W, n=3, m=5, p=7):
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 3, 5, 7
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try to find a lower-rank decomposition using ALS
    for rank in [105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 42, 40, 38, 35, 30, 25, 20]:
        if _elapsed() > TIME_LIMIT - 30:
            break
        
        np.random.seed(42)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        for it in range(50):
            if _elapsed() > TIME_LIMIT - 10:
                break
            
            # Fix V and W, solve for U
            U_new = np.zeros((nm, rank))
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    b = np.ones(p)
                    U_col, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = U_col
            U = U_new
            
            # Fix U and W, solve for V
            V_new = np.zeros((mp, rank))
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    b = np.ones(n)
                    V_col, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = V_col
            V = V_new
            
            # Fix U and V, solve for W
            W_new = np.zeros((pn, rank))
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    b = np.ones(m)
                    W_col, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = W_col
            W = W_new
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list):
            return {'U': U_list, 'V': V_list, 'W': W_list}
    
    # Fallback to schoolbook
    rank = nm * mp * pn
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {'U': U_list, 'V': V_list, 'W': W_list}
