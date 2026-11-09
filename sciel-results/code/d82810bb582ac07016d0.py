import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def _strassen_2x2():
    """
    Return Strassen rank-7 decomposition for 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    U[i*2+j, r] = coeff for A[i,j]
    V[i*2+j, r] = coeff for B[i,j]
    W[j*2+i, r] = coeff for C[i,j]  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    strassen_A = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    strassen_B = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    strassen_C = [
        [((0, 0), 1), ((1, 1), 1)],
        [((1, 0), 1), ((1, 1), -1)],
        [((0, 1), 1), ((1, 1), 1)],
        [((0, 0), 1), ((1, 0), 1)],
        [((0, 0), -1), ((0, 1), 1)],
        [((1, 1), 1)],
        [((0, 0), 1)],
    ]

    for r in range(7):
        for (ai, aj), c in strassen_A[r].items():
            U[ai * 2 + aj, r] += c
        for (bi, bj), c in strassen_B[r].items():
            V[bi * 2 + bj, r] += c
        for (ci, cj), c in strassen_C[r]:
            W[cj * 2 + ci, r] += c

    return U, V, W

def _build_6x6_from_2x2_blocks_strassen():
    """
    Build 6x6 multiplication using 9 blocks of 2x2 multiplication with Strassen (rank 7).
    Total rank = 27 * 7 = 189
    """
    n, m, p = 6, 6, 6
    rank = 189

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U2, V2, W2 = _strassen_2x2()

    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                base_col = (bi * 3 + bj) * 3 + bk
                col_offset = base_col * 7

                for r in range(7):
                    for i in range(2):
                        for j in range(2):
                            local_u_row = i * 2 + j
                            global_u_row = (2 * bi + i) * m + (2 * bj + j)
                            U[global_u_row, col_offset + r] += U2[local_u_row, r]

                    for j in range(2):
                        for k in range(2):
                            local_v_row = j * 2 + k
                            global_v_row = (2 * bj + j) * p + (2 * bk + k)
                            V[global_v_row, col_offset + r] += V2[local_v_row, r]

                    for i in range(2):
                        for k in range(2):
                            local_w_row = k * 2 + i
                            global_w_row = (2 * bk + k) * n + (2 * bi + i)
                            W[global_w_row, col_offset + r] += W2[local_w_row, r]

    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 130

    U, V, W = _build_6x6_from_2x2_blocks_strassen()
    
    return {"U": U, "V": V, "W": W}
