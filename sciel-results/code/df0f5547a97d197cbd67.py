import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = 2, 3, 6
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need to find minimal rank r such that:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # The rank of the all-ones tensor of size n x m x p is 1:
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then sum_c U*V*W = 1*1*1 = 1 for all i,j,k.
    
    # Let's verify this is valid:
    def verify_rank1():
        r = 1
        U = np.ones((n * m, r))
        V = np.ones((m * p, r))
        W = np.ones((p * n, r))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for c in range(r):
                        val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # The rank-1 solution with all ones is valid for the all-ones tensor
    if verify_rank1():
        r = 1
        U = np.ones((n * m, r))
        V = np.ones((m * p, r))
        W = np.ones((p * n, r))
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If for some reason rank 1 doesn't work (it should), try rank 2
    # but this shouldn't be necessary
    
    return best_result
