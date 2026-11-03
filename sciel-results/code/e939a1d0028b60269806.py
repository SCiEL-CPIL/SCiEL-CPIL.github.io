import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Block decomposition: split p=9 into 4+4+1
    # (4,4,4) + (4,4,4) + (4,4,1) = 64 + 64 + 4 = 132 rank (schoolbook for each block)
    
    rank = 132
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Block 1: k=0,1,2,3
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Block 2: k=4,5,6,7
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Block 3: k=8
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1.0
            V[j*p+8, col] = 1.0
            W[8*n+i, col] = 1.0
            col += 1
    
    # Verify
    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to full schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
