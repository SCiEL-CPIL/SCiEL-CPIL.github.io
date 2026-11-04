import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen 2x2 (Rank 7)
    # U2, V2: 4x7. Rows: (0,0), (0,1), (1,0), (1,1)
    # W2: 4x7. Rows: (0,0), (0,1), (1,0), (1,1) corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
    # Note: W index k*n+i. For 2x2, n=2.
    # k=0,i=0 -> 0. k=0,i=1 -> 1. k=1,i=0 -> 2. k=1,i=1 -> 3.
    # So W2 rows: 0:C00, 1:C01, 2:C10, 3:C11.
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # M4: A11(B10-B00) -> C00
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C10, C11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> -C00
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1

    # 2. Winograd 3x3 (Rank 8)
    # U3, V3: 9x8. Rows: i*3+j
    # W3: 9x8. Rows: k*3+i
    
    U3 = np.zeros((9, 8), dtype=int)
    V3 = np.zeros((9, 8), dtype=int)
    W3 = np.zeros((9, 8), dtype=int)
    
    # Based on Inspiration 2
    # p0
    U3[0, 0] = 1; U3[1, 0] = 1; U3[2, 0] = 1
    V3[0, 0] = 1; V3[3, 0] = 1; V3[6, 0] = 1
    W3[0, 0] = 1; W3[3, 0] = 1; W3[6, 0] = 1
    W3[1, 0] = -1; W3[4, 0] = -1; W3[7, 0] = -1
    W3[2, 0] = -1; W3[5, 0] = -1; W3[8, 0] = -1

    # p1
    U3[1, 1] = 1; U3[2, 1] = -1
    V3[0, 1] = 1; V3[3, 1] = -1
    W3[1, 1] = 1; W3[4, 1] = 1; W3[7, 1] = 1
    W3[3, 1] = -1; W3[6, 1] = -1; W3[8, 1] = -1

    # p2
    U3[1, 2] = -1; U3[2, 2] = 1
    V3[2, 2] = 1; V3[5, 2] = -1
    W3[2, 2] = 1; W3[5, 2] = 1; W3[8, 2] = 1
    W3[1, 2] = -1; W3[4, 2] = -1; W3[7, 2] = -1

    # p3
    U3[3, 3] = 1; U3[4, 3] = 1; U3[5, 3] = 1
    V3[0, 3] = 1; V3[3, 3] = 1; V3[6, 3] = 1
    W3[3, 3] = 1; W3[6, 3] = 1; W3[8, 3] = 1
    W3[4, 3] = -1; W3[7, 3] = -1; W3[5, 3] = -1

    # p4
    U3[4, 4] = 1; U3[5, 4] = -1
    V3[0, 4] = 1; V3[3, 4] = -1
    W3[4, 4] = 1; W3[7, 4] = 1; W3[8, 4] = 1
    W3[6, 4] = -1; W3[5, 4] = -1; W3[3, 4] = -1

    # p5
    U3[4, 5] = -1; U3[5, 5] = 1
    V3[2, 5] = 1; V3[5, 5] = -1
    W3[5, 5] = 1; W3[8, 5] = 1; W3[7, 5] = 1
    W3[4, 5] = -1; W3[6, 5] = -1; W3[3, 5] = -1

    # p6
    U3[6, 6] = 1; U3[7, 6] = 1; U3[8, 6] = 1
    V3[0, 6] = 1; V3[3, 6] = 1; V3[6, 6] = 1
    W3[6, 6] = 1; W3[8, 6] = 1; W3[5, 6] = 1
    W3[7, 6] = -1; W3[4, 6] = -1; W3[3, 6] = -1

    # p7
    U3[7, 7] = 1; U3[8, 7] = -1
    V3[0, 7] = 1; V3[3, 7] = -1
    W3[7, 7] = 1; W3[8, 7] = 1; W3[5, 7] = 1
    W3[6, 7] = -1; W3[4, 7] = -1; W3[3, 7] = -1

    # 3. Combine
    rank = 7 * 8
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    for t2 in range(7):
        for t3 in range(8):
            t = t2 * 8 + t3
            
            # U
            for i2 in range(2):
                for j2 in range(2):
                    u2_val = U2[2*i2+j2, t2]
                    if u2_val == 0: continue
                    for i3 in range(3):
                        for j3 in range(3):
                            u3_val = U3[3*i3+j3, t3]
                            if u3_val == 0: continue
                            
                            i_g = 3*i2 + i3
                            j_g = 3*j2 + j3
                            row = i_g * 6 + j_g
                            U[row, t] += u2_val * u3_val

            # V
            for j2 in range(2):
                for k2 in range(2):
                    v2_val = V2[2*j2+k2, t2]
                    if v2_val == 0: continue
                    for j3 in range(3):
                        for k3 in range(3):
                            v3_val = V3[3*j3+k3, t3]
                            if v3_val == 0: continue
                            
                            j_g = 3*j2 + j3
                            k_g = 3*k2 + k3
                            row = j_g * 6 + k_g
                            V[row, t] += v2_val * v3_val

            # W
            for k2 in range(2):
                for i2 in range(2):
                    w2_val = W2[2*k2+i2, t2]
                    if w2_val == 0: continue
                    for k3 in range(3):
                        for i3 in range(3):
                            w3_val = W3[3*k3+i3, t3]
                            if w3_val == 0: continue
                            
                            k_g = 3*k2 + k3
                            i_g = 3*i2 + i3
                            row = k_g * 6 + i_g
                            W[row, t] += w2_val * w3_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
