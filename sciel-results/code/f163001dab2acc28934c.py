import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # The tensor M_{2,2,8} is the direct sum of 4 independent M_{2,2,2} tensors.
    # The minimal rank of M_{2,2,2} is 7 (Strassen).
    # Therefore, the minimal rank of M_{2,2,8} is 4 * 7 = 28.
    # We construct the exact rank-28 decomposition.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for 2x2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[c00, c01], [c10, c11]]
    #
    # P1 = (a+d)(e+h) -> c00 += P1, c11 += P1
    # P2 = (c+d)e     -> c10 += P2, c11 -= P2
    # P3 = a(f-h)     -> c01 += P3, c11 += P3
    # P4 = d(g-e)     -> c00 += P4, c10 += P4
    # P5 = (a+b)h     -> c00 -= P5, c01 += P5
    # P6 = (c-b)(e+f) -> c11 += P6
    # P7 = (b+d)(g+h) -> c00 += P7
    
    strassen_products = [
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
    ]
    
    # Iterate over the 4 blocks of the 2x8 matrix B
    # B is 2x8, so we have 4 blocks of size 2x2
    # Block q corresponds to columns 2q and 2q+1 of B
    for q in range(p // 2):
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r
            
            # Fill U (A indices)
            # U shape is (n*m, rank) = (4, 28)
            # Index in U: i*m + j
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
                
            # Fill V (B indices)
            # V shape is (m*p, rank) = (16, 28)
            # Index in V: j*p + k
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
                
            # Fill W (C indices)
            # W shape is (p*n, rank) = (16, 28)
            # Index in W: k*n + i (Official trace convention W[k*n+i, r])
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
