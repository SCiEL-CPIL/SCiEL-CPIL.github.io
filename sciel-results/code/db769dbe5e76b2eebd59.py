"""Matrix multiplication 2x5x6 with Strassen block decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 2, 5, 6
    rank = n * m * p  # 60, schoolbook upper bound
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition terms
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A10)B11
    # M5 = (A01-A10)(B00+B10)
    # M6 = (A01-A11)(B01+B11)
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M5
    
    # Each entry in the expansion gives (A_idx, B_idx, C_idx) with coefficient
    # A_idx = (local_row, local_col), B_idx = (local_row, local_col), C_idx = (local_row, local_col)
    
    strassen_terms = [
        # (A_r, A_c, B_r, B_c, C_r, C_c, coeff)
        # M0 = (A00+A11)(B00+B11)
        (0, 0, 0, 0, 0, 0, 1),
        (1, 1, 0, 0, 0, 0, 1),
        (0, 0, 1, 1, 0, 0, 1),
        (1, 1, 1, 1, 0, 0, 1),
        # M1 = (A10+A11)B00
        (1, 0, 0, 0, 1, 0, 1),
        (1, 1, 0, 0, 1, 0, 1),
        # M2 = A00(B01-B11)
        (0, 0, 0, 1, 0, 0, 1),
        (0, 0, 1, 1, 0, 0, -1),
        # M3 = A11(B10-B00)
        (1, 1, 1, 0, 1, 0, 1),
        (1, 1, 0, 0, 1, 0, -1),
        # M4 = (A00+A10)B11
        (0, 0, 1, 1, 0, 0, 1),
        (1, 0, 1, 1, 0, 0, 1),
        # M5 = (A01-A10)(B00+B10)
        (0, 1, 0, 0, 1, 0, 1),
        (1, 0, 0, 0, 1, 0, -1),
        (0, 1, 1, 0, 1, 0, 1),
        (1, 0, 1, 0, 1, 0, -1),
        # M6 = (A01-A11)(B01+B11)
        (0, 1, 0, 1, 0, 0, 1),
        (1, 1, 0, 1, 0, 0, -1),
        (0, 1, 1, 1, 0, 0, 1),
        (1, 1, 1, 1, 0, 0, -1),
    ]
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:6]
    # Split B into three 2x2 blocks
    for block in range(3):
        if time.time() > deadline:
            break
        k_offset = block * 2
        for (ar, ac, br, bc, cr, cc, coeff) in strassen_terms:
            g_ar = ar
            g_ac = ac
            g_br = br
            g_bc = k_offset + bc
            g_cr = k_offset + cr
            g_cc = cc
            U[g_ar * m + g_ac][col] += coeff
            V[g_br * p + g_bc][col] += coeff
            W[g_cr * n + g_cc][col] += coeff
            col += 1
    
    # Part 2: A[0:2, 2] * B[2, 0:6]  (2x1x6)
    for i in range(2):
        for k in range(6):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 3: A[0:2, 3:5] * B[3:5, 0:6]
    for block in range(3):
        if time.time() > deadline:
            break
        k_offset = block * 2
        for (ar, ac, br, bc, cr, cc, coeff) in strassen_terms:
            g_ar = ar
            g_ac = 3 + ac
            g_br = 3 + br
            g_bc = k_offset + bc
            g_cr = k_offset + cr
            g_cc = cc
            U[g_ar * m + g_ac][col] += coeff
            V[g_br * p + g_bc][col] += coeff
            W[g_cr * n + g_cc][col] += coeff
            col += 1
    
    # Trim to actual rank used
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    # Verify correctness
    def verify(U_mat, V_mat, W_mat, n, m, p):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        C_actual = np.zeros((p, n))
        U_arr = np.array(U_mat)
        V_arr = np.array(V_mat)
        W_arr = np.array(W_mat)
        for c in range(U_arr.shape[1]):
            C_actual += U_arr[:, c] * V_arr[:, c] * W_arr[:, c].reshape(p, n)
        return np.allclose(C_actual, C_expected, atol=1e-9)
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
