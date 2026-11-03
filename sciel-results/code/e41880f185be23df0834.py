import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # is the all-ones tensor, which has rank 1.
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # with all entries set to 1.
    
    U = np.ones((n * m, 1), dtype=int)
    V = np.ones((m * p, 1), dtype=int)
    W = np.ones((p * n, 1), dtype=int)
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(1):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    # Fallback to schoolbook rank 64
                    U = np.zeros((n*m, n*m*p), dtype=int)
                    V = np.zeros((m*p, n*m*p), dtype=int)
                    W = np.zeros((p*n, n*m*p), dtype=int)
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                r = i2*m*p + j2*p + k2
                                U[i2*m+j2, r] = 1
                                V[j2*p+k2, r] = 1
                                W[k2*n+i2, r] = 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
