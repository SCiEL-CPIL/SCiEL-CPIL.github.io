import time
import numpy as np
import random

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def verify(U, V, W):
        rank = U.shape[1]
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            T += np.einsum('i,j,k->ijk', u, v, w)
            
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    jp = b // p
                    k = b % p
                    kp = c // n
                    ip = c % n
                    if i == ip and j == jp and k == kp:
                        if abs(T[a, b, c] - 1.0) > 1e-6:
                            return False
                    else:
                        if abs(T[a, b, c] - 0.0) > 1e-6:
                            return False
        return True

    def get_strassen_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        strassen_products = [
            {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
            {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
            {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
            {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
            {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
            {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
            {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
        ]
        
        for q in range(p // 2):
            k1 = 2 * q
            for r, prod in enumerate(strassen_products):
                t = q * 7 + r
                for (i, j), coeff in prod['A'].items():
                    U[i * m + j, t] = coeff
                for (j, k_sub), coeff in prod['B'].items():
                    k_global = k1 + k_sub
                    V[j * p + k_global, t] = coeff
                for (i, k_sub), coeff in prod['C'].items():
                    k_global = k1 + k_sub
                    W[k_global * n + i, t] = coeff
        return U, V, W

    # Fallback
    U, V, W = get_strassen_28()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
