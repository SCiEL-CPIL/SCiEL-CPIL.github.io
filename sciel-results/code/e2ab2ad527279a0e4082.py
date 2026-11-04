import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    # 1. Strassen for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: i*m+j, j*p+k, k*n+i
    # n=2, m=2, p=2
    U_strassen = np.array([
        [1, 0, 0, 0, 0, 0, 0], # 0,0
        [0, 1, 1, 0, 0, 0, 0], # 0,1
        [0, 1, 0, 1, 0, 0, 0], # 1,0
        [0, 0, 0, 0, 1, 1, 1]  # 1,1
    ])
    V_strassen = np.array([
        [1, 0, 0, 0, 0, 0, 0], # 0,0
        [0, 1, 0, 0, 0, 0, 0], # 0,1
        [0, 0, 1, 0, 0, 0, 0], # 1,0
        [0, 0, 0, 1, 1, 1, 1]  # 1,1
    ])
    # Wait, Strassen decomposition:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # U (A part):
    # M1: A11, A22
    # M2: A21, A22
    # M3: A11
    # M4: A22
    # M5: A11, A12
    # M6: A21, A11
    # M7: A12, A22
    
    U_s = np.zeros((4, 7))
    U_s[0, 0] = 1 # A11
    U_s[3, 0] = 1 # A22
    U_s[2, 1] = 1 # A21
    U_s[3, 1] = 1 # A22
    U_s[0, 2] = 1 # A11
    U_s[3, 3] = 1 # A22
    U_s[0, 4] = 1 # A11
    U_s[1, 4] = 1 # A12
    U_s[2, 5] = 1 # A21
    U_s[0, 5] = -1 # A11
    U_s[1, 6] = 1 # A12
    U_s[3, 6] = -1 # A22
    
    V_s = np.zeros((4, 7))
    V_s[0, 0] = 1 # B11
    V_s[3, 0] = 1 # B22
    V_s[0, 1] = 1 # B11
    V_s[2, 2] = 1 # B21
    V_s[3, 2] = -1 # B22
    V_s[1, 3] = 1 # B12
    V_s[0, 3] = -1 # B11
    V_s[3, 4] = 1 # B22
    V_s[0, 5] = 1 # B11
    V_s[1, 5] = 1 # B12
    V_s[2, 6] = 1 # B21
    V_s[3, 6] = 1 # B22
    
    W_s = np.zeros((4, 7))
    # C11 = M1 + M4 - M5 - M7 -> idx 0
    W_s[0, 0] = 1
    W_s[0, 3] = 1
    W_s[0, 4] = -1
    W_s[0, 6] = -1
    # C12 = M3 + M5 -> idx 1
    W_s[1, 2] = 1
    W_s[1, 4] = 1
    # C21 = M2 + M4 -> idx 2
    W_s[2, 1] = 1
    W_s[2, 3] = 1
    # C22 = M1 - M2 + M3 + M7 -> idx 3
    W_s[3, 0] = 1
    W_s[3, 1] = -1
    W_s[3, 2] = 1
    W_s[3, 6] = 1

    # 2. T_1,2,2 (Rank 4)
    # n=1, m=2, p=2
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    U_122 = np.array([
        [1, 0, 0, 0], # A0
        [0, 1, 0, 0]  # A1
    ])
    V_122 = np.array([
        [1, 0, 0, 0], # B00
        [0, 0, 1, 0], # B01
        [0, 1, 0, 0], # B10
        [0, 0, 0, 1]  # B11
    ])
    W_122 = np.array([
        [1, 0, 1, 0], # C0
        [0, 1, 0, 1]  # C1
    ])
    # Check:
    # r=0: A0 B00 C0
    # r=1: A1 B10 C0
    # r=2: A0 B01 C1
    # r=3: A1 B11 C1
    # Correct.

    # 3. T_2,4,4 = T_2,2,2 x T_1,2,2
    # Rank 7 * 4 = 28
    # n=2, m=4, p=4
    # U: 8x28
    # V: 16x28
    # W: 8x28
    
    U_244 = np.zeros((8, 28))
    V_244 = np.zeros((16, 28))
    W_244 = np.zeros((8, 28))
    
    for r1 in range(7):
        for r2 in range(4):
            r = r1 * 4 + r2
            # U
            for i in range(2):
                for j in range(2):
                    u_val = U_s[i*2+j, r1]
                    if u_val != 0:
                        # i' = 0, j' = j? No.
                        # T_2,2,2 has i in {0,1}, j in {0,1}.
                        # T_1,2,2 has i'=0, j' in {0,1}.
                        # Product index: i, j = j1*2 + j2?
                        # m=4. j in {0,1,2,3}.
                        # j1 from T_2,2,2 (0,1). j2 from T_1,2,2 (0,1).
                        # j = j1*2 + j2.
                        # U_s index: i*2 + j1.
                        # U_122 index: 0*2 + j2 = j2.
                        for j2 in range(2):
                            u122_val = U_122[j2, r2]
                            if u122_val != 0:
                                j = j1*2 + j2
                                U_244[i*4+j, r] += u_val * u122_val
            # V
            for j1 in range(2):
                for k1 in range(2):
                    v_val = V_s[j1*2+k1, r1]
                    if v_val != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                v122_val = V_122[j2*2+k2, r2]
                                if v122_val != 0:
                                    j = j1*2 + j2
                                    k = k1*2 + k2
                                    V_244[j*4+k, r] += v_val * v122_val
            # W
            for k1 in range(2):
                for i in range(2):
                    w_val = W_s[k1*2+i, r1]
                    if w_val != 0:
                        for k2 in range(2):
                            w122_val = W_122[k2, r2]
                            if w122_val != 0:
                                k = k1*2 + k2
                                W_244[k*2+i, r] += w_val * w122_val

    # 4. T_1,4,4 (Rank 16)
    # n=1, m=4, p=4
    # C_k = sum_j A_j B_jk
    U_144 = np.zeros((4, 16))
    V_144 = np.zeros((16, 16))
    W_144 = np.zeros((4, 16))
    
    for j in range(4):
        for k in range(4):
            r = j*4 + k
            U_144[j, r] = 1
            V_144[j*4+k, r] = 1
            W_144[k, r] = 1

    # 5. T_3,4,4 = T_2,4,4 + T_1,4,4
    # n=3, m=4, p=4
    # Rank 28 + 16 = 44
    U_344 = np.zeros((12, 44))
    V_344 = np.zeros((16, 44))
    W_344 = np.zeros((12, 44))
    
    # First 2 rows of A (from T_2,4,4)
    U_344[0:8, 0:28] = U_244
    V_344[:, 0:28] = V_244
    W_344[0:8, 0:28] = W_244
    
    # 3rd row of A (from T_1,4,4)
    U_344[8:12, 28:44] = U_144
    V_344[:, 28:44] = V_144
    W_344[8:12, 28:44] = W_144

    # 6. T_3,4,8 = T_3,4,4 x T_1,1,2
    # Rank 44
    # n=3, m=4, p=8
    U = np.zeros((12, 44))
    V = np.zeros((32, 44))
    W = np.zeros((24, 44))
    
    # T_1,1,2 is identity-like rank 1
    # U' = 1, V' = [1, 1]^T, W' = [1, 1]^T
    # U_348[i, j, r] = U_344[i, j, r] * 1
    # V_348[j, k, r] = V_344[j, k/2, r] * 1
    # W_348[k, i, r] = W_344[k/2, i, r] * 1
    
    U = U_344
    for r in range(44):
        for j in range(4):
            for k4 in range(4):
                val = V_344[j*4+k4, r]
                if val != 0:
                    k = k4*2 + 0
                    V[j*8+k, r] = val
                    k = k4*2 + 1
                    V[j*8+k, r] = val
        for i in range(3):
            for k4 in range(4):
                val = W_344[k4*2+i, r]
                if val != 0:
                    k = k4*2 + 0
                    W[k*3+i, r] = val
                    k = k4*2 + 1
                    W[k*3+i, r] = val

    # Convert to list of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
