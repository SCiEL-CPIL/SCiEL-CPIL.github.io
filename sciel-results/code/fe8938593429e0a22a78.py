import numpy as np
import time

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # U index: i*m+j = 2*i+j
    # V index: j*p+k = 9*j+k
    # W index: k*n+i = 2*k+i
    
    terms = []
    
    # Strassen blocks for k-pairs: (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        
        # A indices: a=0 (0,0), b=1 (0,1), c=2 (1,0), d=3 (1,1)
        # B indices: e=0 (0,k0), f=1 (0,k1), g=2 (1,k0), h=3 (1,k1) relative to block, 
        # but absolute V indices are:
        # B[0,k0] -> k0
        # B[0,k1] -> k1
        # B[1,k0] -> 9+k0
        # B[1,k1] -> 9+k1
        
        # C indices:
        # C[0,k0] -> 2*k0
        # C[1,k0] -> 2*k0+1
        # C[0,k1] -> 2*k1
        # C[1,k1] -> 2*k1+1
        
        # M1 = (a+d)(e+h)
        U1 = {0: 1, 3: 1}
        V1 = {k0: 1, 9+k1: 1}
        W1 = {2*k0: 1, 2*k1+1: 1}
        terms.append((U1, V1, W1))
        
        # M2 = (c+d)e
        U2 = {2: 1, 3: 1}
        V2 = {k0: 1}
        W2 = {2*k0+1: 1, 2*k1+1: -1}
        terms.append((U2, V2, W2))
        
        # M3 = a(f-h)
        U3 = {0: 1}
        V3 = {k1: 1, 9+k1: -1}
        W3 = {2*k1: 1, 2*k1+1: 1}
        terms.append((U3, V3, W3))
        
        # M4 = d(g-e)
        U4 = {3: 1}
        V4 = {9+k0: 1, k0: -1}
        W4 = {2*k0: 1, 2*k0+1: 1}
        terms.append((U4, V4, W4))
        
        # M5 = (a+b)h
        U5 = {0: 1, 1: 1}
        V5 = {9+k1: 1}
        W5 = {2*k0: -1, 2*k1: 1}
        terms.append((U5, V5, W5))
        
        # M6 = (c-a)(e+f)
        U6 = {2: 1, 0: -1}
        V6 = {k0: 1, k1: 1}
        W6 = {2*k1+1: 1}
        terms.append((U6, V6, W6))
        
        # M7 = (b-d)(g+h)
        U7 = {1: 1, 3: -1}
        V7 = {9+k0: 1, 9+k1: 1}
        W7 = {2*k0: 1}
        terms.append((U7, V7, W7))
        
    # Schoolbook for k=8
    k = 8
    # C[0,8] = a*B[0,8] + b*B[1,8]
    # C[1,8] = c*B[0,8] + d*B[1,8]
    
    # S1 = a * B[0,8]
    terms.append(({0: 1}, {k: 1}, {2*k: 1}))
    # S2 = b * B[1,8]
    terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
    # S3 = c * B[0,8]
    terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
    # S4 = d * B[1,8]
    terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
    
    rank = len(terms)
    
    # Use calculated dimensions to ensure compatibility with evaluator
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
