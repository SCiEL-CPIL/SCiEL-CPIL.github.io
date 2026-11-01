import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    # Schoolbook rank 64 as fallback
    rank_school = n * m * p  # 64
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Rank-56 Strassen decomposition
    # Split j into 2 blocks of 2, k into 4 blocks of 2
    # 8 sub-problems of 2x2 * 2x2, each rank 7
    rank_56 = 56
    U_56 = np.zeros((nm, rank_56))
    V_56 = np.zeros((mp, rank_56))
    W_56 = np.zeros((pn, rank_56))

    # Strassen terms for 2x2 * 2x2 = 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A * B
    #
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M7

    # U coefficients: for each term, coefficients of [a00, a01, a10, a11]
    U_strassen = np.array([
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [-1, 0, 1, 0],  # M6: a10-a00
        [0, 1, 0, -1]   # M7: a01-a11
    ])

    # V coefficients: for each term, coefficients of [b00, b01, b10, b11]
    V_strassen = np.array([
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 1, 0, -1],  # M3: b01-b11
        [-1, 0, 1, 0],  # M4: b10-b00
        [0, 0, 0, 1],   # M5: b11
        [1, 1, 0, 0],   # M6: b00+b01
        [0, 0, 1, 1]    # M7: b10+b11
    ])

    # W coefficients: for each term, coefficients of [C00, C01, C10, C11]
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M7
    W_strassen = np.array([
        [1, 0, 0, 1],   # M1: C00+, C11+
        [0, 0, 1, -1],  # M2: C10+, C11-
        [0, 1, 0, 1],   # M3: C01+, C11+
        [-1, 1, 0, 0],  # M4: C00-, C01+
        [1, 0, 1, 0],   # M5: C00+, C10+
        [0, 0, 0, 0],   # M6: not used in output
        [-1, 0, 0, 1]   # M7: C00-, C11+
    ])

    col = 0
    for a_block in range(2):  # j in {0,1} or {2,3}
        for b_block in range(4):  # k in {0,1}, {2,3}, {4,5}, {6,7}
            for s in range(7):
                # U: A[i, j] where i in {0,1}, j in {2*a_block, 2*a_block+1}
                # Flat index: i*m + j = i*4 + j
                # U_strassen[s] = [coeff for A[0,2ab], coeff for A[0,2ab+1],
                #                   coeff for A[1,2ab], coeff for A[1,2ab+1]]
                U_56[0 * m + a_block * 2 + 0, col] = U_strassen[s, 0]
                U_56[0 * m + a_block * 2 + 1, col] = U_strassen[s, 1]
                U_56[1 * m + a_block * 2 + 0, col] = U_strassen[s, 2]
                U_56[1 * m + a_block * 2 + 1, col] = U_strassen[s, 3]

                # V: B[j, k] where j in {2*a_block, 2*a_block+1}, k in {2*b_block, 2*b_block+1}
                # Flat index: j*p + k = j*8 + k
                # V_strassen[s] = [coeff for B[2ab,2bb], coeff for B[2ab,2bb+1],
                #                   coeff for B[2ab+1,2bb], coeff for B[2ab+1,2bb+1]]
                V_56[(a_block * 2 + 0) * p + b_block * 2 + 0, col] = V_strassen[s, 0]
                V_56[(a_block * 2 + 0) * p + b_block * 2 + 1, col] = V_strassen[s, 1]
                V_56[(a_block * 2 + 1) * p + b_block * 2 + 0, col] = V_strassen[s, 2]
                V_56[(a_block * 2 + 1) * p + b_block * 2 + 1, col] = V_strassen[s, 3]

                # W: C[i, k] where i in {0,1}, k in {2*b_block, 2*b_block+1}
                # Flat index: k*n + i = k*2 + i
                # W_strassen[s] = [coeff for C[0,2bb], coeff for C[0,2bb+1],
                #                   coeff for C[1,2bb], coeff for C[1,2bb+1]]
                W_56[(b_block * 2 + 0) * n + 0, col] = W_strassen[s, 0]
                W_56[(b_block * 2 + 0) * n + 1, col] = W_strassen[s, 1]
                W_56[(b_block * 2 + 1) * n + 0, col] = W_strassen[s, 2]
                W_56[(b_block * 2 + 1) * n + 1, col] = W_strassen[s, 3]

                col += 1

    # Verify the rank-56 decomposition
    U_t = U_56.T.reshape(n, m, rank_56)
    V_t = V_56.T.reshape(m, p, rank_56)
    W_t = W_56.T.reshape(p, n, rank_56)

    result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    target = np.ones((n, m, p))

    if np.allclose(result, target, atol=1e-10):
        return {"U": U_56.tolist(), "V": V_56.tolist(), "W": W_56.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
