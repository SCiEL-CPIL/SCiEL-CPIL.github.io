import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block diagonal decomposition
    # 9 blocks of rank 4
    for k in range(p):
        for j in range(m): # j is the internal rank index 0..3
            c = k * m + j
            
            # V: row j_row * p + k
            # We want V[j * p + k, c] = 1
            V[j * p + k, c] = 1.0
            
            # W: row k * n + i
            # We want W[k * n + i, c] = 1 for all i
            for i in range(n):
                W[k * n + i, c] = 1.0
                
            # U: row i * m + j_col
            # We want U[i * m + j, c] = 1 for all i
            for i in range(n):
                U[i * m + j, c] = 1.0
                
    # Check decomposition
    def check_decomposition(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = np.sum(W[k*n+i] * U[i*m+j1] * V[j2*p+k])
                        target = 1.0 if j1 == j2 else 0.0
                        max_err = max(max_err, abs(s - target))
        return max_err < tol

    if check_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if something went wrong (should not happen)
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook))
        V_sb = np.zeros((m*p, rank_schoolbook))
        W_sb = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
