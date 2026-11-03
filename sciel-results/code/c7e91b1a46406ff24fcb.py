import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n*m*p = 3*4*7 = 84
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices for schoolbook multiplication
    # The tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # For schoolbook, each term in the sum corresponds to a unique (i,j,k) tuple.
    # We map (i,j,k) to a rank index r.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = col
                if r >= rank:
                    continue
                
                # U index: (i, j) -> row i*m + j
                u_idx = i * m + j
                # V index: (j, k) -> row j*p + k
                v_idx = j * p + k
                # W index: (k, i) -> row k*n + i (following W[k*n+i, r] convention)
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                col += 1
                
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                    
    return {"U": U, "V": V, "W": W}
