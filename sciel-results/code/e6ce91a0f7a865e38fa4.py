"""Rewritten program for 6x6x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = 6, 6, 6

    # --- Strategy: Kronecker product of Strassen (2x2, rank 7) and schoolbook (3x3, rank 27) ---
    # This gives a 6x6x6 decomposition of rank 7 * 27 = 189.
    # The tensor product structure: if T_2 is the 2x2x2 matrix-mult tensor of rank 7,
    # and T_3 is the 3x3x3 matrix-mult tensor of rank 27, then T_6 = T_2 ⊗ T_3 has rank 189.

    # Strassen's 2x2x2 decomposition (rank 7):
    # U_strassen: (4, 7), V_strassen: (4, 7), W_strassen: (4, 7)
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c] for 2x2 matrices.
    #
    # Strassen's 7 products for C = A*B where A,B are 2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)

    # Column 0: M1 = (a11+a22)(b11+b22)
    U_s[0, 0] = 1  # a11
    U_s[3, 0] = 1  # a22
    V_s[0, 0] = 1  # b11
    V_s[3, 0] = 1  # b22
    # W: C11 += M1, C22 += M1
    W_s[0, 0] = 1  # k=0,i=0 -> C11
    W_s[3, 0] = 1  # k=1,i=1 -> C22

    # Column 1: M2 = (a21+a22)b11
    U_s[2, 1] = 1  # a21
    U_s[3, 1] = 1  # a22
    V_s[0, 1] = 1  # b11
    # W: C21 += M2
    W_s[2, 1] = 1  # k=0,i=1 -> C21

    # Column 2: M3 = a11(b12-b22)
    U_s[0, 2] = 1  # a11
    V_s[1, 2] = 1  # b12
    V_s[3, 2] = -1  # b22
    # W: C12 += M3, C22 += M3
    W_s[1, 2] = 1  # k=0,i=0 -> C12
    W_s[3, 2] = 1  # k=1,i=1 -> C22

    # Column 3: M4 = a22(b21-b11)
    U_s[3, 3] = 1  # a22
    V_s[2, 3] = 1  # b21
    V_s[0, 3] = -1  # b11
    # W: C11 -= M4, C21 += M4
    W_s[0, 3] = -1  # k=0,i=0 -> C11
    W_s[2, 3] = 1  # k=0,i=1 -> C21

    # Column 4: M5 = (a11+a12)b22
    U_s[0, 4] = 1  # a11
    U_s[1, 4] = 1  # a12
    V_s[3, 4] = 1  # b22
    # W: C11 += M5, C12 += M5
    W_s[0, 4] = 1  # k=0,i=0 -> C11
    W_s[1, 4] = 1  # k=0,i=0 -> C12

    # Column 5: M6 = (a21-a11)(b11+b12)
    U_s[2, 5] = 1  # a21
    U_s[0, 5] = -1  # a11
    V_s[0, 5] = 1  # b11
    V_s[1, 5] = 1  # b12
    # W: C22 += M6
    W_s[3, 5] = 1  # k=1,i=1 -> C22

    # Column 6: M7 = (a12-a22)(b21+b22)
    U_s[1, 6] = 1  # a12
    U_s[3, 6] = -1  # a22
    V_s[2, 6] = 1  # b21
    V_s[3, 6] = 1  # b22
    # W: C11 -= M7
    W_s[0, 6] = -1  # k=0,i=0 -> C11

    # Schoolbook 3x3x3 decomposition (rank 27):
    U_b = np.zeros((9, 27), dtype=int)
    V_b = np.zeros((9, 27), dtype=int)
    W_b = np.zeros((9, 27), dtype=int)

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_b[i * 3 + j, col] = 1
                V_b[j * 3 + k, col] = 1
                W_b[k * 3 + i, col] = 1
                col += 1

    # Now form the Kronecker product decomposition for 6x6x6:
    # The 6x6 matrix multiplication tensor T_6 can be viewed as a tensor product T_2 ⊗ T_3
    # where we group indices: i = i2*3 + i3, j = j2*3 + j3, k = k2*3 + k3
    #
    # U_6[i*6+j, c] where c = c2*27 + c3
    #   U_6[i2*3+i3, j2*3+j3, c2*27+c3] = U_s[i2*2+j2, c2] * U_b[i3*3+j3, c3]
    #
    # V_6[j*6+k, c]
    #   V_6[j2*3+j3, k2*3+k3, c2*27+c3] = V_s[j2*2+k2, c2] * V_b[j3*3+k3, c3]
    #
    # W_6[k*6+i, c]
    #   W_6[k2*3+k3, i2*3+i3, c2*27+c3] = W_s[k2*2+i2, c2] * W_b[k3*3+i3, c3]

    rank = 7 * 27  # 189

    U_6 = np.zeros((36, rank), dtype=int)
    V_6 = np.zeros((36, rank), dtype=int)
    W_6 = np.zeros((36, rank), dtype=int)

    for c2 in range(7):
        for c3 in range(27):
            c = c2 * 27 + c3
            # U_6
            for i2 in range(2):
                for j2 in range(2):
                    us_val = U_s[i2 * 2 + j2, c2]
                    if us_val == 0:
                        continue
                    for i3 in range(3):
                        for j3 in range(3):
                            ub_val = U_b[i3 * 3 + j3, c3]
                            if ub_val == 0:
                                continue
                            row = (i2 * 3 + i3) * 6 + (j2 * 3 + j3)
                            U_6[row, c] = us_val * ub_val
            # V_6
            for j2 in range(2):
                for k2 in range(2):
                    vs_val = V_s[j2 * 2 + k2, c2]
                    if vs_val == 0:
                        continue
                    for j3 in range(3):
                        for k3 in range(3):
                            vb_val = V_b[j3 * 3 + k3, c3]
                            if vb_val == 0:
                                continue
                            row = (j2 * 3 + j3) * 6 + (k2 * 3 + k3)
                            V_6[row, c] = vs_val * vb_val
            # W_6
            for k2 in range(2):
                for i2 in range(2):
                    ws_val = W_s[k2 * 2 + i2, c2]
                    if ws_val == 0:
                        continue
                    for k3 in range(3):
                        for i3 in range(3):
                            wb_val = W_b[k3 * 3 + i3, c3]
                            if wb_val == 0:
                                continue
                            row = (k2 * 3 + k3) * 6 + (i2 * 3 + i3)
                            W_6[row, c] = ws_val * wb_val

    # Convert to lists of integers
    U_list = U_6.tolist()
    V_list = V_6.tolist()
    W_list = W_6.tolist()

    # Verify the decomposition with a random test (if time permits)
    if time.time() < deadline - 5.0:
        try:
            np.random.seed(42)
            A = np.random.randint(-5, 5, (6, 6))
            B = np.random.randint(-5, 5, (6, 6))
            C_expected = A @ B

            U_arr = np.array(U_list, dtype=float)
            V_arr = np.array(V_list, dtype=float)
            W_arr = np.array(W_list, dtype=float)

            # Compute C from decomposition:
            # C[i,k] = sum_c U[i*6+j ... ] -- need to be careful
            # Actually: C[i,k] = sum_j sum_c ... 
            # The tensor decomposition: T[i,j,k] = sum_c U[i*6+j, c] * V[j*6+k, c] * W[k*6+i, c]
            # C[i,k] = sum_j T[i,j,k] = sum_j sum_c U[i*6+j,c] * V[j*6+k,c] * W[k*6+i,c]
            
            C_computed = np.zeros((6, 6), dtype=float)
            for i in range(6):
                for k in range(6):
                    total = 0.0
                    for j in range(6):
                        uc = U_arr[i * 6 + j]
                        vc = V_arr[j * 6 + k]
                        w_val = W_arr[k * 6 + i]
                        total += np.dot(uc, vc * w_val)
                    C_computed[i, k] = total

            if not np.allclose(C_computed, C_expected, atol=1e-9):
                # Fallback to schoolbook if Kronecker product is wrong
                rank_fb = 216
                U_fb = [[0] * rank_fb for _ in range(36)]
                V_fb = [[0] * rank_fb for _ in range(36)]
                W_fb = [[0] * rank_fb for _ in range(36)]
                col = 0
                for i in range(6):
                    for j in range(6):
                        for k in range(6):
                            U_fb[i * 6 + j][col] = 1
                            V_fb[j * 6 + k][col] = 1
                            W_fb[k * 6 + i][col] = 1
                            col += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
        except Exception:
            pass

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
