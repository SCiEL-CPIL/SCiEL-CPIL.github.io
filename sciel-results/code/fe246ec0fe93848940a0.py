import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        U_i = U[i * m:(i + 1) * m, :]
        for k in range(p):
            V_k = V[k::p, :]
            w = W[k * n + i, :]
            M = U_i * w[None, :] @ V_k.T
            if np.max(np.abs(M - np.eye(m))) > tol:
                return False
    return True

def _strassen_2x2():
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-d)(f+g)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(e+g)
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    
    if time.time() > _DEADLINE:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 56 construction
    U2, V2, W2 = _strassen_2x2()
    
    # Build 4x4x4
    rank4 = 56
    U4 = np.zeros((16, rank4))
    V4 = np.zeros((16, rank4))
    W4 = np.zeros((16, rank4))
    
    col = 0
    for ci in range(2):
        for cj in range(2):
            for bk in range(2):
                for s in range(7):
                    for di in range(2):
                        for dj in range(2):
                            U4[(2*ci+di)*4 + (2*bk+dj), col] = U2[di*2+dj, s]
                            V4[(2*bk+di)*4 + (2*cj+dj), col] = V2[di*2+dj, s]
                            W4[(2*cj+dj)*4 + (2*ci+di), col] = W2[di*2+dj, s]
                    col += 1
    
    # Build 4x4x8
    rank = 2 * rank4
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    U[:, 0:rank4] = U4
    U[:, rank4:2*rank4] = U4
    
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:rank4] = V4[j*4 + k, :]
            V[j*8 + (k+4), rank4:2*rank4] = V4[j*4 + k, :]
            
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:rank4] = W4[k*4 + i, :]
            W[(k+4)*4 + i, rank4:2*rank4] = W4[k*4 + i, :]
            
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
