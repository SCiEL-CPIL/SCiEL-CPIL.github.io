import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # s: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict (i,j) -> val
    # v_coeffs: dict (j,k) -> val
    # w_coeffs: dict (k,i) -> val
    
    # s=0
    # u: (0,0)=1, (1,1)=1
    # v: (0,0)=1, (1,1)=1
    # w: (0,0)=1, (1,1)=1
    # s=1
    # u: (1,0)=1, (1,1)=1
    # v: (0,0)=1
    # w: (1,0)=1, (1,1)=-1
    # s=2
    # u: (0,0)=1
    # v: (0,1)=1, (1,1)=-1
    # w: (0,1)=1, (1,1)=1
    # s=3
    # u: (1,1)=1
    # v: (1,0)=1, (0,0)=-1
    # w: (0,0)=1, (1,0)=1
    # s=4
    # u: (0,0)=1, (0,1)=1
    # v: (1,1)=1
    # w: (0,0)=-1, (0,1)=1
    # s=5
    # u: (1,0)=1, (0,0)=-1
    # v: (0,0)=1, (0,1)=1
    # w: (1,1)=1
    # s=6
    # u: (0,1)=1, (1,1)=-1
    # v: (1,0)=1, (1,1)=1
    # w: (0,0)=1

    strassen_terms = [
        { 'u': {(0,0): 1, (1,1): 1}, 'v': {(0,0): 1, (1,1): 1}, 'w': {(0,0): 1, (1,1): 1} },
        { 'u': {(1,0): 1, (1,1): 1}, 'v': {(0,0): 1}, 'w': {(1,0): 1, (1,1): -1} },
        { 'u': {(0,0): 1}, 'v': {(0,1): 1, (1,1): -1}, 'w': {(0,1): 1, (1,1): 1} },
        { 'u': {(1,1): 1}, 'v': {(1,0): 1, (0,0): -1}, 'w': {(0,0): 1, (1,0): 1} },
        { 'u': {(0,0): 1, (0,1): 1}, 'v': {(1,1): 1}, 'w': {(0,0): -1, (0,1): 1} },
        { 'u': {(1,0): 1, (0,0): -1}, 'v': {(0,0): 1, (0,1): 1}, 'w': {(1,1): 1} },
        { 'u': {(0,1): 1, (1,1): -1}, 'v': {(1,0): 1, (1,1): 1}, 'w': {(0,0): 1} },
    ]
    
    for b in range(3):
        for s in range(7):
            c = b * 7 + s
            term = strassen_terms[s]
            
            # Fill U
            for (i, j), val in term['u'].items():
                U[i * m + j][c] += val
                
            # Fill V
            for (j, k_local), val in term['v'].items():
                k = 2 * b + k_local
                V[j * p + k][c] += val
                
            # Fill W
            for (k_local, i), val in term['w'].items():
                k = 2 * b + k_local
                W[k * n + i][c] += val
                
    return {"U": U, "V": V, "W": W}
