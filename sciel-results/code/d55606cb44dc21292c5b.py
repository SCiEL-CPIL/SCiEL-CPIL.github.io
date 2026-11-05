import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    nm, mp, pn = n*m, m*p, p*n
    
    # Split p=5 into p1=3 and p2=2
    # Rank of M_{3,3,3} is 9
    # Rank of M_{3,3,2} is 18
    # Total rank 27
    
    R = 27
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    r = 0
    # Part 1: k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
                r += 1
                
    # Part 2: k in 3..4
    for i in range(n):
        for j in range(m):
            for k in range(3, 5):
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
