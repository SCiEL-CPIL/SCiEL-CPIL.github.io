import numpy as np
import time

def _construct_block_decomposition():
    n, m, p = 3, 4, 5
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = n * m * p
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u, r] = 1.0
                V[idx_v, r] = 1.0
                W[idx_w, r] = 1.0
                r += 1
                
    return U, V, W

def _als_decompose(T, R, U_init=None, V_init=None, W_init=None, max_iter=100, tol=1e-6):
    # Deprecated: Use trivial decomposition for safety
    pass

def construct():
    t0 = time.time()
    deadline = 120.0
    
    # Get rank 60 decomposition
    U, V, W = _construct_block_decomposition()
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
