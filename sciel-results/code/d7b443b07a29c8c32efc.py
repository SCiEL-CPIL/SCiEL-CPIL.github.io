import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find a low-rank decomposition
    for rank in range(28, 57):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(42 + rank)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for iteration in range(100):
            if time.time() > deadline - 5:
                break
            
            # Solve for W
            for i in range(n):
                for k in range(p):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i, :] = w
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p+k, :] = v
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i*m+j, :] = u
        
        # Check error
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1)**2
        
        if error < 1e-6:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Schoolbook fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
