import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    # Rank 8 decomposition
    # T_{i,j,k} = 1 for all i,j,k
    # Decomposition: sum_{i',j'} e_{i',j'} \otimes e_{j', \cdot} \otimes e_{\cdot, i'}
    # R = n * m = 8
    
    rank = n * m
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j, r] = 1
            for k in range(p):
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                
    if verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
