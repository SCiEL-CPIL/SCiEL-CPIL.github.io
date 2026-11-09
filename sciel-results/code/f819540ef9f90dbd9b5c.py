"""Program for rank-minimal tensor decomposition of 2x4 x 4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = 2, 4, 8
    deadline = time.time() + 130  # safety margin before 150s kill

    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # For C = A * B where A, B, C are 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M6 - M7

    # Strassen factors for 2x2x2:
    strassen_U = [
        [[1,0],[0,1]],   # M1: a, d
        [[0,0],[1,1]],   # M2: c, d
        [[1,0],[0,0]],   # M3: a
        [[0,0],[0,1]],   # M4: d
        [[1,1],[0,0]],   # M5: a, b
        [[-1,0],[1,0]],  # M6: -a, c
        [[0,1],[0,-1]],  # M7: b, -d
    ]
    strassen_V = [
        [[1,0],[0,1]],   # M1: e, h
        [[1,0],[0,0]],   # M2: e
        [[0,1],[0,-1]],  # M3: f, -h
        [[-1,0],[1,0]],  # M4: -e, g
        [[0,0],[0,1]],   # M5: h
        [[1,1],[0,0]],   # M6: e, f
        [[0,0],[1,1]],   # M7: g, h
    ]
    strassen_W = [
        [[1,0],[0,1]],   # M1: C00, C11
        [[0,1],[0,0]],   # M2: C10 (W[j,i]: j=0,i=1)
        [[0,1],[0,0]],   # M3: C01 (W[j,i]: j=1,i=0)
        [[-1,1],[0,0]],  # M4: -C00, C10
        [[1,1],[0,0]],   # M5: C00, C01
        [[0,0],[0,1]],   # M6: C11 (W[j,i]: j=1,i=1)
        [[-1,0],[0,-1]], # M7: -C00, -C11
    ]

    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for col_group in range(4):
        for row_part in range(2):
            for r in range(7):
                for i in range(2):
                    for k in range(2):
                        u_val = strassen_U[r][i][k]
                        if u_val != 0:
                            k_global = k + 2 * row_part
                            U[i * m + k_global, col] += u_val
                for k in range(2):
                    for j in range(2):
                        v_val = strassen_V[r][k][j]
                        if v_val != 0:
                            k_global = k + 2 * row_part
                            j_global = col_group * 2 + j
                            V[k_global * p + j_global, col] += v_val
                for j in range(2):
                    for i in range(2):
                        w_val = strassen_W[r][j][i]
                        if w_val != 0:
                            j_global = col_group * 2 + j
                            W[j_global * n + i, col] += w_val
                col += 1

    # Verify with a few random matrices
    def verify(U_mat, V_mat, W_mat, n, m, p, seed=42, atol=1e-9):
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                total = 0
                for r in range(rank):
                    u_sum = 0
                    for k in range(m):
                        u_sum += U_mat[i*m + k, r] * A[i, k]
                    v_sum = 0
                    for k in range(m):
                        v_sum += V_mat[k*p + j, r] * B[k, j]
                    total += W_mat[j*n + i, r] * u_sum * v_sum
                C_computed[i, j] = total
        
        return np.allclose(C_computed, C_expected, atol=atol)

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook rank 64
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

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
