"""Program for rank-minimal tensor decomposition of 2x4 x 4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = 2, 4, 8
    deadline = time.time() + 140  # safety margin

    # Standard Strassen decomposition for 2x2x2
    strassen_U = [
        [[1, 0], [0, 1]],   # M1
        [[0, 0], [1, 1]],   # M2
        [[1, 0], [0, 0]],   # M3
        [[0, 0], [0, 1]],   # M4
        [[1, 1], [0, 0]],   # M5
        [[-1, 0], [1, 0]],  # M6
        [[0, 1], [0, -1]]   # M7
    ]
    strassen_V = [
        [[1, 0], [0, 1]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[-1, 0], [1, 0]],  # M3: B10 - B00
        [[0, 1], [0, -1]],  # M4: B01 - B10
        [[0, 0], [0, 1]],   # M5
        [[1, 1], [0, 0]],   # M6
        [[0, 0], [1, 1]]    # M7
    ]
    strassen_W = [
        [[1, 0], [0, 1]],   # M1: C00, C11
        [[0, 1], [0, 0]],   # M2: C10
        [[0, 0], [1, 0]],   # M3: C01
        [[-1, 0], [0, 1]],  # M4: -C00, C10
        [[1, 0], [1, 0]],   # M5: C00, C01
        [[0, 0], [0, 1]],   # M6: C11
        [[-1, 0], [0, -1]]  # M7: -C00, -C11
    ]

    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for col_group in range(4):
        for row_part in range(2):
            for r in range(7):
                # U
                for i in range(2):
                    for k in range(2):
                        u_val = strassen_U[r][i][k]
                        if u_val != 0:
                            k_global = k + 2 * row_part
                            U[i * m + k_global, col] += u_val
                
                # V
                for k in range(2):
                    for j in range(2):
                        v_val = strassen_V[r][k][j]
                        if v_val != 0:
                            k_global = k + 2 * row_part
                            j_global = col_group * 2 + j
                            V[k_global * p + j_global, col] += v_val
                
                # W
                for j in range(2):
                    for i in range(2):
                        w_val = strassen_W[r][j][i]
                        if w_val != 0:
                            j_global = col_group * 2 + j
                            W[j_global * n + i, col] += w_val
                
                col += 1

    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p):
        U_arr = np.array(U_mat)
        V_arr = np.array(V_mat)
        W_arr = np.array(W_mat)
        rank_val = U_arr.shape[1]
        
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                total = 0
                for r in range(rank_val):
                    u_sum = 0
                    for k in range(m):
                        u_sum += U_arr[i*m + k, r] * A[i, k]
                    v_sum = 0
                    for k in range(m):
                        v_sum += V_arr[k*p + j, r] * B[k, j]
                    total += W_arr[j*n + i, r] * u_sum * v_sum
                C_computed[i, j] = total
        
        return np.allclose(C_computed, C_expected, atol=1e-9)

    if not verify(U, V, W, n, m, p):
        # Fallback
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k, col] = 1
                    V[k*p + j, col] = 1
                    W[j*n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
