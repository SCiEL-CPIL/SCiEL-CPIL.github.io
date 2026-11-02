import numpy as np
import time
from itertools import product, combinations
from scipy.linalg import lstsq

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Rank is n*m? No, rank is n * (min(m,p)?) 
    # For 1 x m x p, rank is min(m,p)? 
    # My derivation: Rank of 1 x n x n is n.
    # Here m=5, p=5. So rank per row i is 5.
    # Total rank n * 5 = 10.
    
    rank = n * m  # 10
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j, r] = 1.0
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                
    # Verify
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
                
    T = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T, target, atol=1e-9):
        # Fallback to schoolbook if construction fails (should not happen)
        rank_school = n * m * p
        U_s = np.zeros((dim_u, rank_school))
        V_s = np.zeros((dim_v, rank_school))
        W_s = np.zeros((dim_w, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
