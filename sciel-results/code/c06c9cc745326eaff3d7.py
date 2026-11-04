import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-nmp decomposition:
    # Each non-zero entry of the tensor is a separate rank-1 term.
    # Target tensor T has T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) (using prompt's W index k*n+i)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                U[a, idx] = 1
                V[b, idx] = 1
                W[c, idx] = 1
                
                idx += 1
                
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
