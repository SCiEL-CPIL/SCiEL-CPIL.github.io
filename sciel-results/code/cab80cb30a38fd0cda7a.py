import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        U[0, col] = 1; U[3, col] = 1
        V[0, col] = 1; V[3, col] = 1
        W[k0, col] = 1; W[k1+2, col] = 1
        col += 1
        
        U[3, col] = 1; U[4, col] = 1
        V[0, col] = 1
        W[k0+2, col] = 1; W[k1+2, col] = -1
        col += 1
        
        U[0, col] = 1
        V[1, col] = 1; V[4, col] = -1
        W[k1, col] = 1; W[k1+2, col] = 1
        col += 1
        
        U[4, col] = 1
        V[3, col] = 1; V[0, col] = -1
        W[k0, col] = 1; W[k0+2, col] = 1
        col += 1
        
        U[0, col] = 1; U[1, col] = 1
        V[4, col] = 1
        W[k0, col] = -1; W[k1, col] = 1
        col += 1
        
        U[3, col] = 1; U[0, col] = -1
        V[0, col] = 1; V[1, col] = 1
        W[k1+2, col] = 1
        col += 1
        
        U[1, col] = 1; U[4, col] = -1
        V[3, col] = 1; V[4, col] = 1
        W[k0, col] = 1
        col += 1
        
        U[2, col] = 1
        V[6+k0, col] = 1
        W[k0, col] = 1
        col += 1
        
        U[2, col] = 1
        V[6+k1, col] = 1
        W[k1, col] = 1
        col += 1
        
        U[5, col] = 1
        V[6+k0, col] = 1
        W[k0+2, col] = 1
        col += 1
        
        U[5, col] = 1
        V[6+k1, col] = 1
        W[k1+2, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
