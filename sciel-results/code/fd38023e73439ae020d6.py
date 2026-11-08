import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 45
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1_term(U_vec, V_vec, W_vec):
        nonlocal col
        if col >= rank:
            return
        for i, val in enumerate(U_vec):
            if val != 0:
                U[i][col] += val
        for i, val in enumerate(V_vec):
            if val != 0:
                V[i][col] += val
        for i, val in enumerate(W_vec):
            if val != 0:
                W[i][col] += val
        col += 1

    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i

    # 1. A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # Schoolbook: 3*3 = 9 terms
    for i in range(3):
        for j in range(3):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(i, j)] = 1
            V_term[v_idx(j, 0)] = 1 # Wait, B is 3x3, so j is row, k is col.
            # B[0:3, 0:3] means rows 0-2, cols 0-2.
            # v_idx(j, k) where j is row of B, k is col of B.
            # Here k is 0-2.
            # So we need a loop for k too.
            pass
            
    # Correct Schoolbook for 3x3x3:
    # for i in 3, j in 3, k in 3:
    # U[i,j], V[j,k], W[k,i]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
                U_term[u_idx(i, j)] = 1
                V_term[v_idx(j, k)] = 1
                W_term[w_idx(i, k)] = 1
                add_rank1_term(U_term, V_term, W_term)

    # 2. A[0:3, 0:3] * B[0:3, 3:7] -> C[0:3, 3:7]
    # Schoolbook: 3*3*4 = 36 terms
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
                U_term[u_idx(i, j)] = 1
                V_term[v_idx(j, k)] = 1
                W_term[w_idx(i, k)] = 1
                add_rank1_term(U_term, V_term, W_term)

    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_true = A_test @ B_test
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += U_arr[i*m + j, r] * A_test[i, j] * V_arr[j*p + k, r] * B_test[j, k] * W_arr[k*n + i, r]
            C_comp[i, k] = val
            
    if not np.allclose(C_comp, C_true, atol=1e-9):
        # Fallback to schoolbook 63
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
