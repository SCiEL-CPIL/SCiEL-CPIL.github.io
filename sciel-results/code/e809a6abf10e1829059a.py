import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = 2, 2, 10
    nm, mp, pn = n * m, m * p, p * n

    def build_rank20():
        rank = 20
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        
        # 4 terms for each column k
        # Term 1: A[0,0] * B[0,k] * C[0,k]
        # Term 2: A[0,1] * B[1,k] * C[0,k]
        # Term 3: A[1,0] * B[0,k] * C[1,k]
        # Term 4: A[1,1] * B[1,k] * C[1,k]
        
        col = 0
        for k in range(p):
            # Term 1
            U[0, col] = 1 # A[0,0]
            V[0*p + k, col] = 1 # B[0,k]
            W[k*n + 0, col] = 1 # C[0,k]
            col += 1
            
            # Term 2
            U[1, col] = 1 # A[0,1]
            V[1*p + k, col] = 1 # B[1,k]
            W[k*n + 0, col] = 1 # C[0,k]
            col += 1
            
            # Term 3
            U[2, col] = 1 # A[1,0]
            V[0*p + k, col] = 1 # B[0,k]
            W[k*n + 1, col] = 1 # C[1,k]
            col += 1
            
            # Term 4
            U[3, col] = 1 # A[1,1]
            V[1*p + k, col] = 1 # B[1,k]
            W[k*n + 1, col] = 1 # C[1,k]
            col += 1
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best = build_rank20()
    
    return best
