import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 2, 9

    # Decomposition strategy:
    # Split the 2x9 matrix B into four 2x2 blocks (cols 0-7) and one 2x1 block (col 8).
    # Use Strassen (rank 7) for each 2x2 block multiplication.
    # Use trivial multiplication (rank 4) for the 2x1 block.
    # Total rank = 4 * 7 + 4 = 32.

    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    current_col = 0

    def add_strassen_block(k_start, k_end):
        nonlocal current_col
        k0 = k_start
        k1 = k_end

        # 1. M1: (A00 + A11) * (B00 + B11)
        U[0][current_col] = 1.0
        U[3][current_col] = 1.0
        V[0*p + k0][current_col] = 1.0
        V[1*p + k1][current_col] = 1.0
        W[k0*n + 0][current_col] = 1.0
        W[k1*n + 1][current_col] = 1.0
        current_col += 1

        # 2. M2: (A10 + A11) * B00
        U[2][current_col] = 1.0
        U[3][current_col] = 1.0
        V[0*p + k0][current_col] = 1.0
        W[k0*n + 1][current_col] = 1.0
        W[k1*n + 1][current_col] = -1.0
        current_col += 1

        # 3. M3: A00 * (B01 - B11)
        U[0][current_col] = 1.0
        V[0*p + k1][current_col] = 1.0
        V[1*p + k1][current_col] = -1.0
        W[k1*n + 0][current_col] = 1.0
        W[k1*n + 1][current_col] = 1.0
        current_col += 1

        # 4. M4: A11 * (B10 - B00)
        U[3][current_col] = 1.0
        V[1*p + k0][current_col] = 1.0
        V[0*p + k0][current_col] = -1.0
        W[k0*n + 0][current_col] = 1.0
        W[k0*n + 1][current_col] = 1.0
        current_col += 1

        # 5. M5: (A00 + A01) * B11
        U[0][current_col] = 1.0
        U[1][current_col] = 1.0
        V[1*p + k1][current_col] = 1.0
        W[k0*n + 0][current_col] = -1.0
        W[k1*n + 0][current_col] = 1.0
        current_col += 1

        # 6. M6: (A10 - A00) * (B00 + B01)
        U[2][current_col] = 1.0
        U[0][current_col] = -1.0
        V[0*p + k0][current_col] = 1.0
        V[0*p + k1][current_col] = 1.0
        W[k1*n + 1][current_col] = 1.0
        current_col += 1

        # 7. M7: (A01 - A11) * (B10 + B11)
        U[1][current_col] = 1.0
        U[3][current_col] = -1.0
        V[1*p + k0][current_col] = 1.0
        V[1*p + k1][current_col] = 1.0
        W[k0*n + 0][current_col] = 1.0
        current_col += 1

    def add_trivial_col(k):
        nonlocal current_col
        # Term 1: A00 * B0k -> C0k
        U[0][current_col] = 1.0
        V[0*p + k][current_col] = 1.0
        W[k*n + 0][current_col] = 1.0
        current_col += 1

        # Term 2: A01 * B1k -> C0k
        U[1][current_col] = 1.0
        V[1*p + k][current_col] = 1.0
        W[k*n + 0][current_col] = 1.0
        current_col += 1

        # Term 3: A10 * B0k -> C1k
        U[2][current_col] = 1.0
        V[0*p + k][current_col] = 1.0
        W[k*n + 1][current_col] = 1.0
        current_col += 1

        # Term 4: A11 * B1k -> C1k
        U[3][current_col] = 1.0
        V[1*p + k][current_col] = 1.0
        W[k*n + 1][current_col] = 1.0
        current_col += 1

    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    add_strassen_block(6, 7)
    add_trivial_col(8)

    # Verification
    def verify(U_list, V_list, W_list):
        U_arr = np.array(U_list, dtype=np.float64)
        V_arr = np.array(V_list, dtype=np.float64)
        W_arr = np.array(W_list, dtype=np.float64)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = U_arr[i*m + k]
                    v_vec = V_arr[k*p + j]
                    w_vec = W_arr[j*n + i]
                    term = u_vec * v_vec * w_vec
                    val = np.sum(term)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    if time.time() < deadline - 5.0:
        err = verify(U, V, W)
        if err > 1e-9:
            rank_t = n * m * p
            U_t = [[0.0] * rank_t for _ in range(n * m)]
            V_t = [[0.0] * rank_t for _ in range(m * p)]
            W_t = [[0.0] * rank_t for _ in range(p * n)]
            c = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_t[ii*m + jj][c] = 1.0
                        V_t[jj*p + kk][c] = 1.0
                        W_t[kk*n + ii][c] = 1.0
                        c += 1
            return {"U": U_t, "V": V_t, "W": W_t}

    return {"U": U, "V": W if False else V, "W": W}
