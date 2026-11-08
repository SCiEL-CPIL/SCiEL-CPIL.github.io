import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    t0 = time.time()
    deadline = t0 + 100
    
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    rank_sb = n*m*p
    U = np.zeros((n*m, rank_sb))
    V = np.zeros((m*p, rank_sb))
    W = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    best = (U.copy(), V.copy(), W.copy(), rank_sb)
    
    T_1 = T.reshape(n*m, m*p*p*n)
    T_2 = np.transpose(T, (1,0,2)).reshape(m*p, n*m*p*n)
    T_3 = np.transpose(T, (2,0,1)).reshape(p*n, n*m*m*p)
    
    def als(U, V, W, max_iter=40):
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            M = (V[:, :, None] * W[None, :, :]).reshape(m*p*p*n, -1)
            G = M.T @ M
            try:
                U = T_1 @ M.T @ np.linalg.inv(G)
            except:
                U = T_1 @ M.T @ np.linalg.pinv(G)
            
            M2 = (U[:, :, None] * W[None, :, :]).reshape(n*m*p*n, -1)
            G2 = M2.T @ M2
            try:
                V = T_2 @ M2.T @ np.linalg.inv(G2)
            except:
                V = T_2 @ M2.T @ np.linalg.pinv(G2)
            
            M3 = (U[:, :, None] * V[None, :, :]).reshape(n*m*m*p, -1)
            G3 = M3.T @ M3
            try:
                W = T_3 @ M3.T @ np.linalg.inv(G3)
            except:
                W = T_3 @ M3.T @ np.linalg.pinv(G3)
        return U, V, W
    
    def err(U, V, W):
        return np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T).max()
    
    rng = np.random.default_rng(42)
    
    for rank in [20, 22, 24, 26, 28, 30, 32]:
        if time.time() > deadline - 15:
            break
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            U = rng.standard_normal((n*m, rank))
            V = rng.standard_normal((m*p, rank))
            W = rng.standard_normal((p*n, rank))
            U, V, W = als(U, V, W, max_iter=40)
            e = err(U, V, W)
            if e < 1e-5:
                # Try rounding and re-optimizing
                Ur, Vr, Wr = np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2
                U2, V2, W2 = als(Ur, Vr, Wr, max_iter=20)
                e2 = err(U2, V2, W2)
                Ur2, Vr2, Wr2 = np.round(U2*2)/2, np.round(V2*2)/2, np.round(W2*2)/2
                e3 = err(Ur2, Vr2, Wr2)
                if e3 < 1e-8:
                    best = (Ur2, Vr2, Wr2, rank)
                    break
                elif e2 < 1e-7:
                    best = (U2, V2, W2, rank)
                    break
                elif e < 1e-7:
                    best = (U, V, W, rank)
                    break
        if best[3] < rank_sb:
            break
    
    U, V, W, r = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
