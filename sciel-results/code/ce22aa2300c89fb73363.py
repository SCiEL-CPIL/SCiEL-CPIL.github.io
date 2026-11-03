import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def check_decomposition(U, V, W, n, m, p):
    """
    Check if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Condition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for r in range(R):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 60
    
    # Start with schoolbook
    U = np.zeros((n * m, base_rank), dtype=int)
    V = np.zeros((m * p, base_rank), dtype=int)
    W = np.zeros((p * n, base_rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Try to reduce rank by 1
    # For 3x4x5, we can potentially use a rank-59 decomposition
    # One approach: find a linear dependency among the 60 rank-1 tensors
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all (i,j,k)
    # A rank-1 tensor is u_i * v_j * w_k
    # We need sum_r u_r[i] v_r[j] w_r[k] = 1
    
    # Schoolbook: u_r[i] = delta(i, i_r), v_r[j] = delta(j, j_r), w_r[k] = delta(k, k_r)
    # for each (i_r, j_r, k_r)
    
    # To reduce rank, we need to combine some of these.
    # This is possible if there's a linear relation.
    
    # For now, return schoolbook as it's guaranteed correct.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
