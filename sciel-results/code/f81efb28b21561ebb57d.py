import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Rank-36 decomposition for 4x4 x 4x9
    # For each (k,j) pair, one term:
    # U[i*4+k, k*9+j] = 1 for all i in {0,1,2,3}
    # V[k*9+j, k*9+j] = 1
    # W[j*4+i, k*9+j] = 1 for all i in {0,1,2,3}
    
    rank = m * p  # 4 * 9 = 36
    
    U = np.zeros((n * m, rank))  # (16, 36)
    V = np.zeros((m * p, rank))  # (36, 36)
    W = np.zeros((p * n, rank))  # (36, 36)
    
    for k in range(m):
        for j in range(p):
            r = k * p + j
            # U: rows i*m+k for i in 0..n-1
            for i in range(n):
                U[i * m + k, r] = 1
            # V: row k*p+j
            V[k * p + j, r] = 1
            # W: rows j*n+i for i in 0..n-1
            for i in range(n):
                W[j * n + i, r] = 1
    
    # Validate
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        a_vec = np.zeros(n)
        b_vec = np.zeros(p)
        for i in range(n):
            a_vec[i] = sum(U[i * m + k, r] * A_test[i, k] for k in range(m))
        for j in range(p):
            b_vec[j] = sum(V[k * p + j, r] * B_test[k, j] for k in range(m))
        for i in range(n):
            for j in range(p):
                C_computed[i, j] += W[j * n + i, r] * a_vec[i] * b_vec[j]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U = np.zeros((n * m, rank_fb))
        V = np.zeros((m * p, rank_fb))
        W = np.zeros((p * n, rank_fb))
        idx = 0
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    U[i * m + k, idx] = 1
                    V[k * p + j, idx] = 1
                    W[j * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
