import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 140

    def verify(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u = U_arr[:, r].reshape(n, m)
            v = V_arr[:, r].reshape(m, p)
            w = W_arr[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m+j, j*p+k, k*n+i] += u[i,j] * v[j,k] * w[k,i]
        
        Texp = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Texp[i*m+j, j*p+k, k*n+i] = 1.0
        
        return np.allclose(T, Texp, atol=1e-9)

    # Strassen 2x2x2 factor matrices
    strassen_U = [
        [1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
        [1,1,0,0],[-1,0,1,0],[0,1,0,-1],
    ]
    strassen_V = [
        [1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],
        [0,0,0,1],[1,1,0,0],[0,0,1,1],
    ]
    strassen_W = [
        [1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],
        [-1,0,1,0],[0,0,0,1],[1,0,0,0],
    ]
    
    rank = 4*7 + 8  # 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for j0 in (0, 2):
        for k0 in (0, 2):
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U[li*m + (j0+lj)][col] = strassen_U[r][li*2+lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(j0+lj)*p + (k0+lk)][col] = strassen_V[r][lj*2+lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0+lk)*n + li][col] = strassen_W[r][lk*2+li]
                col += 1
    
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 40
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, column] = 1
                V_school[j*p+k, column] = 1
                W_school[k*n+i, column] = 1
                column += 1
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
