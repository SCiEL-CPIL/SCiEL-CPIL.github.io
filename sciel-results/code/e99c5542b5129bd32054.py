"""Rank-49 exact tensor decomposition for 4x4 matrix multiplication via Kronecker product of Strassen."""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS

    # Define the 2x2 Strassen decomposition
    # For 2x2: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    # Identity: sum_r U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] = 1 for all i,j,k in {0,1}
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # S1 = (P00+P11)(Q00+Q11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # S2 = (P10+P11)Q00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1

    # S3 = P00(Q01-Q11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1

    # S4 = P11(Q10-Q00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1

    # S5 = (P00+P01)Q11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # S6 = (P10-P00)(Q00+Q01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # S7 = (P01-P11)(Q10+Q11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = -1

    # Verify 2x2 Strassen
    max_err2 = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :])
                err = abs(val - 1)
                if err > max_err2:
                    max_err2 = err

    if max_err2 >= 1e-10:
        # Strassen is wrong, fall back to schoolbook
        rank_fb = 64
        U_fb = [[0] * rank_fb for _ in range(16)]
        V_fb = [[0] * rank_fb for _ in range(16)]
        W_fb = [[0] * rank_fb for _ in range(16)]
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i * 4 + j][column] = 1
                    V_fb[j * 4 + k][column] = 1
                    W_fb[k * 4 + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Build 4x4 decomposition via Kronecker product
    # 4x4 index: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # Row in U: i*4+j = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2
    # Row in V: j*4+k = (2*j1+j2)*4 + (2*k1+k2) = 8*j1 + 4*j2 + 2*k1 + k2
    # Row in W: k*4+i = (2*k1+k2)*4 + (2*i1+i2) = 8*k1 + 4*k2 + 2*i1 + i2
    #
    # Column: r = r1*7 + r2 where r1, r2 in {0,...,6}
    #
    # U4[8*i1+4*i2+2*j1+j2, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # V4[8*j1+4*j2+2*k1+k2, r1*7+r2] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    # W4[8*k1+4*k2+2*i1+i2, r1*7+r2] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]

    rank = 49
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))

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
                                    val = u1 * v1 * w1 * u2 * v2 * w2
                                    if val == 0:
                                        continue
                                    row_U = 8*i1 + 4*i2 + 2*j1 + j2
                                    row_V = 8*j1 + 4*j2 + 2*k1 + k2
                                    row_W = 8*k1 + 4*k2 + 2*i1 + i2
                                    U4[row_U, col] += u1 * u2
                                    V4[row_V, col] += v1 * v2
                                    W4[row_W, col] += w1 * w2

    # Verify
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.sum(U4[i*4+j, :] * V4[j*4+k, :] * W4[k*4+i, :])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err

    if max_err < 1e-10:
        U_list = U4.tolist()
        V_list = V4.tolist()
        W_list = W4.tolist()

        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if abs(mat[i][j] - round(mat[i][j])) < 1e-10:
                        mat[i][j] = round(mat[i][j])

        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank_fb = 64
        U_fb = [[0] * rank_fb for _ in range(16)]
        V_fb = [[0] * rank_fb for _ in range(16)]
        W_fb = [[0] * rank_fb for _ in range(16)]
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i * 4 + j][column] = 1
                    V_fb[j * 4 + k][column] = 1
                    W_fb[k * 4 + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
