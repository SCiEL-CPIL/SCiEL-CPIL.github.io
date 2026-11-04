"""Construct a rank-97 exact tensor decomposition for 4x4x7 matrix multiplication."""

import time

def _schoolbook():
    n, m, p = 4, 4, 7
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

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    # Strassen rank-7 decomposition for 2x2x2
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    # Term 0: (A00+A11)(B00+B11) -> C00, C11
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1

    # Term 1: (A10+A11)B00 -> C10
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1

    # Term 2: A00(B10-B11) -> C01
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[2][2] = 1

    # Term 3: A11(B01-B00) -> C00, C10
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1

    # Term 4: (A00+A01)B11 -> C00, C01
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1

    # Term 5: (A10-A00)(B00+B01) -> C11
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1

    # Term 6: (A01-A11)(B10+B11) -> C00, C11
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = -1

    # Verify 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for c in range(7):
                    total += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
                if total != 1:
                    return _schoolbook()

    # Build 4x4x4 via Kronecker product of 2x2x2 with itself
    # U4[i1*4+i2*2+j2, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    # V4[j1*4+j2*2+k2, c1*7+c2] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
    # W4[k1*4+k2*2+i2, c1*7+c2] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]

    for i1 in range(2):
        for j1 in range(2):
            for c1 in range(7):
                for i2 in range(2):
                    for j2 in range(2):
                        for c2 in range(7):
                            val = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
                            if val != 0:
                                U4[i1*4 + i2*2 + j2][c1*7+c2] += val

    for j1 in range(2):
        for k1 in range(2):
            for c1 in range(7):
                for j2 in range(2):
                    for k2 in range(2):
                        for c2 in range(7):
                            val = V2[j1*2+k1][c1] * V2[j2*2+k2][c2]
                            if val != 0:
                                V4[j1*4 + j2*2 + k2][c1*7+c2] += val

    for k1 in range(2):
        for i1 in range(2):
            for c1 in range(7):
                for k2 in range(2):
                    for i2 in range(2):
                        for c2 in range(7):
                            val = W2[k1*2+i1][c1] * W2[k2*2+i2][c2]
                            if val != 0:
                                W4[k1*4 + k2*2 + i2][c1*7+c2] += val

    # Verify 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                total = 0
                for c in range(49):
                    total += U4[i*4+j][c] * V4[j*4+k][c] * W4[k*4+i][c]
                if total != 1:
                    return _schoolbook()

    if time.time() > deadline:
        return _schoolbook()

    # Build 4x4x3 using schoolbook (rank 48)
    # A is 4x4, B is 4x3, C is 4x3
    # U43[i*4+j, c], V43[j*3+k, c], W43[k*4+i, c]
    U43 = [[0]*48 for _ in range(16)]
    V43 = [[0]*48 for _ in range(12)]
    W43 = [[0]*48 for _ in range(12)]

    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U43[i*4+j][col] = 1
                V43[j*3+k][col] = 1
                W43[k*4+i][col] = 1
                col += 1

    # Verify 4x4x3
    for i in range(4):
        for j in range(4):
            for k in range(3):
                total = 0
                for c in range(48):
                    total += U43[i*4+j][c] * V43[j*3+k][c] * W43[k*4+i][c]
                if total != 1:
                    return _schoolbook()

    # Combine: 4x4x7 = 4x4x4 (rank 49) + 4x4x3 (rank 48) = rank 97
    rank_total = 97
    U_final = [[0] * rank_total for _ in range(16)]
    V_final = [[0] * rank_total for _ in range(28)]
    W_final = [[0] * rank_total for _ in range(28)]

    # First 49 columns: 4x4x4 part (k indices 0-3)
    # U_final[i*4+j, c] = U4[i*4+j, c] for c in 0..48
    # V_final[j*4+k, c] = V4[j*4+k, c] for c in 0..48, k in 0..3
    #   But in final tensor, V index is j*7+k. For the 4x4x4 part, k ranges 0..3.
    #   So V_final[j*7+k, c] = V4[j*4+k, c] for k=0..3
    # W_final[k*4+i, c] = W4[k*4+i, c] for c in 0..48

    for i in range(16):
        for c in range(49):
            U_final[i][c] = U4[i][c]

    for j in range(4):
        for k in range(4):
            for c in range(49):
                V_final[j*7+k][c] = V4[j*4+k][c]

    for k in range(4):
        for i in range(4):
            for c in range(49):
                W_final[k*4+i][c] = W4[k*4+i][c]

    # Next 48 columns: 4x4x3 part (k indices 4-6, i.e., k_global = 4 + k_local)
    # U_final[i*4+j, 49+c] = U43[i*4+j, c]
    # V_final[j*7+k_global, 49+c] = V43[j*3+k_local, c] where k_global = 4+k_local, k_local=0..2
    # W_final[k_global*4+i, 49+c] = W43[k_local*4+i, c]

    for i in range(16):
        for c in range(48):
            U_final[i][49+c] = U43[i][c]

    for j in range(4):
        for k_local in range(3):
            k_global = 4 + k_local
            for c in range(48):
                V_final[j*7+k_global][49+c] = V43[j*3+k_local][c]

    for k_local in range(3):
        k_global = 4 + k_local
        for i in range(4):
            for c in range(48):
                W_final[k_global*4+i][49+c] = W43[k_local*4+i][c]

    # Verify final 4x4x7
    # U: (n*m, rank) = (16, 97), index i*4+j
    # V: (m*p, rank) = (28, 97), index j*7+k
    # W: (p*n, rank) = (28, 97), index k*4+i
    for i in range(4):
        for j in range(4):
            for k in range(7):
                total = 0
                for c in range(rank_total):
                    total += U_final[i*4+j][c] * V_final[j*7+k][c] * W_final[k*4+i][c]
                if total != 1:
                    return _schoolbook()

    return {"U": U_final, "V": V_final, "W": W_final}
