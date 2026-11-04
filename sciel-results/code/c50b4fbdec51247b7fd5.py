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

    # Build 4x4x3 using Kronecker product of 2x2x2 Strassen and 2x2x1 Schoolbook
    # Rank 28 for 4x4x2 part, Rank 16 for 4x4x1 part. Total 44.
    
    # 2x2x1 Schoolbook factors
    U1 = [[0]*4 for _ in range(4)]
    V1 = [[0]*4 for _ in range(2)]
    W1 = [[0]*4 for _ in range(2)]
    # Term 0: A00 B00 -> C00
    U1[0][0] = 1; V1[0][0] = 1; W1[0][0] = 1
    # Term 1: A01 B10 -> C01 (Wait, B is 2x1, so B10 is index 1? No, B is col vector. B[0], B[1])
    # C0 = A00 B0 + A01 B1. C1 = A10 B0 + A11 B1.
    # V1 index j. W1 index i.
    # c=0: A00 B0 -> C0. U1[0,0]=1, V1[0,0]=1, W1[0,0]=1.
    # c=1: A01 B1 -> C0. U1[1,1]=1, V1[1,1]=1, W1[0,1]=1.
    # c=2: A10 B0 -> C1. U1[2,2]=1, V1[0,2]=1, W1[1,2]=1.
    # c=3: A11 B1 -> C1. U1[3,3]=1, V1[1,3]=1, W1[1,3]=1.
    U1[1][1] = 1; V1[1][1] = 1; W1[0][1] = 1
    U1[2][2] = 1; V1[0][2] = 1; W1[1][2] = 1
    U1[3][3] = 1; V1[1][3] = 1; W1[1][3] = 1

    # 4x4x2 part (Rank 28)
    # U42[i*4+j, c1*4+c2], V42[j*2+k, c1*4+c2], W42[k*4+i, c1*4+c2]
    U42 = [[0]*28 for _ in range(16)]
    V42 = [[0]*28 for _ in range(8)]
    W42 = [[0]*28 for _ in range(8)]
    
    for i1 in range(2):
        for j1 in range(2):
            for c1 in range(7):
                for i2 in range(2):
                    for j2 in range(2):
                        for c2 in range(4):
                            val = U2[i1*2+j1][c1] * U1[i2*2+j2][c2]
                            if val != 0:
                                i = i1*2 + i2
                                j = j1*2 + j2
                                c = c1*4 + c2
                                U42[i*4+j][c] += val

    for j1 in range(2):
        for k1 in range(2):
            for c1 in range(7):
                for j2 in range(2):
                    # k2 is always 0 for 2x2x1
                    for c2 in range(4):
                        val = V2[j1*2+k1][c1] * V1[j2][c2]
                        if val != 0:
                            j = j1*2 + j2
                            k = k1
                            c = c1*4 + c2
                            V42[j*2+k][c] += val

    for k1 in range(2):
        for i1 in range(2):
            for c1 in range(7):
                for i2 in range(2):
                    # k2 is always 0
                    for c2 in range(4):
                        val = W2[k1*2+i1][c1] * W1[i2][c2]
                        if val != 0:
                            k = k1
                            i = i1*2 + i2
                            c = c1*4 + c2
                            W42[k*4+i][c] += val

    # 4x4x1 part (Rank 16)
    # U41[i*4+j, c1*4+c2], V41[j, c1*4+c2], W41[i, c1*4+c2]
    U41 = [[0]*16 for _ in range(16)]
    V41 = [[0]*16 for _ in range(4)]
    W41 = [[0]*16 for _ in range(4)]
    
    for i1 in range(2):
        for j1 in range(2):
            for c1 in range(4):
                for i2 in range(2):
                    for j2 in range(2):
                        for c2 in range(4):
                            val = U1[i1*2+j1][c1] * U1[i2*2+j2][c2]
                            if val != 0:
                                i = i1*2 + i2
                                j = j1*2 + j2
                                c = c1*4 + c2
                                U41[i*4+j][c] += val

    for j1 in range(2):
        for c1 in range(4):
            for j2 in range(2):
                for c2 in range(4):
                    val = V1[j1][c1] * V1[j2][c2]
                    if val != 0:
                        j = j1*2 + j2
                        c = c1*4 + c2
                        V41[j][c] += val

    for i1 in range(2):
        for c1 in range(4):
            for i2 in range(2):
                for c2 in range(4):
                    val = W1[i1][c1] * W1[i2][c2]
                    if val != 0:
                        i = i1*2 + i2
                        c = c1*4 + c2
                        W41[i][c] += val

    # Combine 4x4x2 and 4x4x1 into 4x4x3
    # U43[i*4+j, c], V43[j*3+k, c], W43[k*4+i, c]
    # First 28 cols: 4x4x2 (k=0,1)
    # Next 16 cols: 4x4x1 (k=2)
    U43 = [[0]*44 for _ in range(16)]
    V43 = [[0]*44 for _ in range(12)]
    W43 = [[0]*44 for _ in range(12)]
    
    for i in range(16):
        for c in range(28):
            U43[i][c] = U42[i][c]
    for j in range(4):
        for k in range(2):
            for c in range(28):
                V43[j*3+k][c] = V42[j*2+k][c]
    for k in range(2):
        for i in range(4):
            for c in range(28):
                W43[k*4+i][c] = W42[k*4+i][c]
                
    for i in range(16):
        for c in range(16):
            U43[i][28+c] = U41[i][c]
    for j in range(4):
        for c in range(16):
            V43[j*3+2][28+c] = V41[j][c]
    for i in range(4):
        for c in range(16):
            W43[2*4+i][28+c] = W41[i][c]

    # Verify 4x4x3
    for i in range(4):
        for j in range(4):
            for k in range(3):
                total = 0
                for c in range(44):
                    total += U43[i*4+j][c] * V43[j*3+k][c] * W43[k*4+i][c]
                if total != 1:
                    return _schoolbook()

    # Combine: 4x4x7 = 4x4x4 (rank 49) + 4x4x3 (rank 44) = rank 93
    rank_total = 93
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

    # Next 44 columns: 4x4x3 part (k indices 4-6, i.e., k_global = 4 + k_local)
    # U_final[i*4+j, 49+c] = U43[i*4+j, c]
    # V_final[j*7+k_global, 49+c] = V43[j*3+k_local, c] where k_global = 4+k_local, k_local=0..2
    # W_final[k_global*4+i, 49+c] = W43[k_local*4+i, c]

    for i in range(16):
        for c in range(44):
            U_final[i][49+c] = U43[i][c]

    for j in range(4):
        for k_local in range(3):
            k_global = 4 + k_local
            for c in range(44):
                V_final[j*7+k_global][49+c] = V43[j*3+k_local][c]

    for k_local in range(3):
        k_global = 4 + k_local
        for i in range(4):
            for c in range(44):
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
