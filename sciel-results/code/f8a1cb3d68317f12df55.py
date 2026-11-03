"""Matrix multiplication 3,3,5 - rank-minimal tensor decomposition."""
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 150.0

def construct():
    n, m, p = 3, 3, 5
    
    # Schoolbook construction: rank 45
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    U = np.zeros((9, 45), dtype=int)
    V = np.zeros((15, 45), dtype=int)
    W = np.zeros((15, 45), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(5):
                U[i*3+j][col] = 1
                V[j*5+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
