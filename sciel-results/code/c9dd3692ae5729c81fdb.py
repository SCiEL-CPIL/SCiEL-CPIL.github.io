import numpy as np

def construct():
    n, m, p = 4, 4, 5
    I, J, K = n*m, m*p, p*n
    
    T = np.zeros((I, J, K))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    best_rank = n * m * p
    best = None
    
    for rank in range(55, best_rank + 1):
        np.random.seed(rank)
        U = np.random.randn(I, rank)
        V = np.random.randn(J, rank)
        W = np.random.randn(K, rank)
        
        for _ in range(30):
            for r in range(rank):
                T1 = T.reshape(I, -1)
                res = T1.copy()
                for r2 in range(rank):
                    if r2 != r:
                        res -= np.outer(U[:, r2], np.outer(V[:, r2], W[:, r2]).ravel())
                vw = np.outer(V[:, r], W[:, r]).ravel()
                U[:, r] = res @ vw / (vw @ vw + 1e-12)
            
            for r in range(rank):
                T2 = T.reshape(J, -1)
                res = T2.copy()
                for r2 in range(rank):
                    if r2 != r:
                        res -= np.outer(V[:, r2], np.outer(W[:, r2], U[:, r2]).ravel())
                wu = np.outer(W[:, r], U[:, r]).ravel()
                V[:, r] = res @ wu / (wu @ wu + 1e-12)
            
            for r in range(rank):
                T3 = T.reshape(K, -1)
                res = T3.copy()
                for r2 in range(rank):
                    if r2 != r:
                        res -= np.outer(W[:, r2], np.outer(U[:, r2], V[:, r2]).ravel())
                uv = np.outer(U[:, r], V[:, r]).ravel()
                W[:, r] = res @ uv / (uv @ uv + 1e-12)
        
        Trec = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(Trec, T, atol=1e-5):
            U2 = np.round(U * 2) / 2
            V2 = np.round(V * 2) / 2
            W2 = np.round(W * 2) / 2
            Trec2 = np.einsum('ir,jr,kr->ijk', U2, V2, W2)
            if np.allclose(Trec2, T, atol=1e-5):
                return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
