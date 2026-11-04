import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the matrix multiplication tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i * m + j
                idx2 = j * p + k
                idx3 = k * n + i
                T[idx1, idx2, idx3] = 1.0
    
    def verify(U, V, W):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-8)
    
    # Rank 30 decomposition: one term per (j,k) pair
    rank = m * p  # 5 * 6 = 30
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                idx1 = i * m + j
                idx2 = j * p + k
                idx3 = k * n + i
                U[idx1, col] = 1.0
                V[idx2, col] = 1.0
                W[idx3, col] = 1.0
            col += 1
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx1 = i * m + j
                    idx2 = j * p + k
                    idx3 = k * n + i
                    U[idx1, col] = 1.0
                    V[idx2, col] = 1.0
                    W[idx3, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
