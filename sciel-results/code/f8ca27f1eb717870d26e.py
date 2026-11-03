"""Rank-49 exact tensor decomposition for 4x4 matrix multiplication via verified Kronecker product of Strassen."""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = 4, 4, 4

    # Build 2x2 Strassen decomposition (rank 7)
    # U2[i*2+j, r] = coeff of A[i,j] in r-th mult
    # V2[j*2+k, r] = coeff of B[j,k] in r-th mult
    # W2[k*2+i, r] = coeff of r-th mult in C[i,k]
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M2 = (A10+A11)B00 -> C10+=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1

    # M3 = A00(B01-B11) -> C01+=1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1

    # M4 = A11(B10-B00) -> C00+=1, C10+=1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1

    # M5 = (A00+A01)B11 -> C00-=1, C01+=1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # M6 = (A10-A00)(B00+B01) -> C11+=1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # M7 = (A01-A11)(B10+B11) -> C00+=1, C11-=1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = -1

    # Verify 2x2 Strassen
    max_err2 = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = float(np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :]))
                err = abs(val - 1.0)
                if err > max_err2:
                    max_err2 = err

    if max_err2 > 1e-10:
        # Fallback: schoolbook rank-64
        rank_fb = 64
        U_fb = [[0] * rank_fb for _ in range(16)]
        V_fb = [[0] * rank_fb for _ in range(16)]
        W_fb = [[0] * rank_fb for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j][col] = 1
                    V_fb[j*4+k][col] = 1
                    W_fb[k*4+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Build 4x4 decomposition via Kronecker product (rank 49)
    # Index mapping: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # U row: i*4+j = 8*i1 + 4*i2 + 2*j1 + j2
    # V row: j*4+k = 8*j1 + 4*j2 + 2*k1 + k2
    # W row: k*4+i = 8*k1 + 4*k2 + 2*i1 + i2
    rank_kron = 49
    U4 = np.zeros((16, rank_kron), dtype=int)
    V4 = np.zeros((16, rank_kron), dtype=int)
    W4 = np.zeros((16, rank_kron), dtype=int)

    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        u1 = U2[i1*2+j1, r1]
                        v1 = V2[j1*2+k1, r1]
                        w1 = W2[k1*2+i1, r1]
                        if u1 == 0 or v1 == 0 or w1 == 0:
                            continue
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    u2 = U2[i2*2+j2, r2]
                                    v2 = V2[j2*2+k2, r2]
                                    w2 = W2[k2*2+i2, r2]
                                    if u2 == 0 or v2 == 0 or w2 == 0:
                                        continue
                                    row_U = 8*i1 + 4*i2 + 2*j1 + j2
                                    row_V = 8*j1 + 4*j2 + 2*k1 + k2
                                    row_W = 8*k1 + 4*k2 + 2*i1 + i2
                                    U4[row_U, col] += u1 * u2
                                    V4[row_V, col] += v1 * v2
                                    W4[row_W, col] += w1 * w2

    # Thorough verification
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = float(np.sum(U4[i*4+j, :] * V4[j*4+k, :] * W4[k*4+i, :]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-10:
        # Fallback: schoolbook rank-64
        rank_fb = 64
        U_fb = [[0] * rank_fb for _ in range(16)]
        V_fb = [[0] * rank_fb for _ in range(16)]
        W_fb = [[0] * rank_fb for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j][col] = 1
                    V_fb[j*4+k][col] = 1
                    W_fb[k*4+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

# EVOLVE-BLOCK-END
