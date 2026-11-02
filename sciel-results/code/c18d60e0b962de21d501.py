"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    r = m  # rank = 5, which is optimal since rank >= m
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # U[i*m+j, c] = 1 if c == j, else 0
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    
    # V[j*p+k, c] = 1 if c == j, else 0
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    
    # W[k*n+i, c] = 1 for all c
    for k in range(p):
        for i in range(n):
            for c in range(r):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p, rank):
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_random_decomp(n, m, p, r):
    """Try to find a random decomposition of rank r."""
    vals = [0, 0.5, 1, -0.5, -1, 0.25, -0.25, 2, -2]
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        u_nonzero = random.sample(range(n * m), min(3, n * m))
        for idx in u_nonzero:
            U[idx, c] = random.choice(vals)
        
        v_nonzero = random.sample(range(m * p), min(5, m * p))
        for idx in v_nonzero:
            V[idx, c] = random.choice(vals)
        
        w_nonzero = random.sample(range(p * n), min(3, p * n))
        for idx in w_nonzero:
            W[idx, c] = random.choice(vals)
    
    if _verify(n, m, p, U, V, W, r):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _verify(n, m, p, U, V, W, r):
    """Check if U,V,W of rank r exactly decompose the matmul tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += W[k * n + i, c] * U[i * m + j, c] * V[j * p + k, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
