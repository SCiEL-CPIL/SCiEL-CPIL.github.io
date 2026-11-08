import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    for rank in range(70, 97):
        if time.time() > deadline:
            break
        
        for trial in range(30):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            for it in range(30):
                if time.time() > deadline:
                    break
                
                for i in range(n):
                    for k in range(p):
                        M = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                        b = np.ones(m)
                        w, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        W[k * n + i, :] = w
                
                for j in range(m):
                    for k in range(p):
                        M = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                        b = np.ones(n)
                        v, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        V[j * p + k, :] = v
                
                for i in range(n):
                    for j in range(m):
                        M = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                        b = np.ones(p)
                        u, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        U[i * m + j, :] = u
                
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-10:
                    break
            
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-8:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
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
