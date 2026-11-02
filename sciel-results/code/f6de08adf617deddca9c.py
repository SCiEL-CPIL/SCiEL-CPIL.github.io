import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2x2
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen indices
    # M1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11? No, W2[k*2+i]
    # C00 gets M1, C11 gets M1.
    # W2 index k*2+i.
    # C00 is i=0, k=0 -> idx 0.
    # C11 is i=1, k=1 -> idx 3.
    W2[0, 0] = 1
    W2[3, 0] = 1
    
    # M2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C10 (i=1, k=0) -> idx 2
    W2[3, 1] = -1 # C11 (i=1, k=1) -> idx 3
    
    # M3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C01 (i=0, k=1) -> idx 1
    W2[3, 2] = 1 # C11 (i=1, k=1) -> idx 3
    
    # M4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1 # C00
    W2[2, 3] = 1 # C10
    
    # M5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C00
    W2[1, 4] = 1 # C01
    
    # M6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # M7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00
    
    # Construct 4x4x4 factors
    # Rank 49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i in range(4):
        for j in range(4):
            I1, I2 = divmod(i, 2)
            J1, J2 = divmod(j, 2)
            row_u = i * 4 + j
            for r_out in range(7):
                for r_in in range(7):
                    r = r_out * 7 + r_in
                    U4[row_u, r] = U2[I1 * 2 + J1, r_out] * U2[I2 * 2 + J2, r_in]
                    
    for j in range(4):
        for k in range(4):
            J1, J2 = divmod(j, 2)
            K1, K2 = divmod(k, 2)
            row_v = j * 4 + k
            for r_out in range(7):
                for r_in in range(7):
                    r = r_out * 7 + r_in
                    V4[row_v, r] = V2[J1 * 2 + K1, r_out] * V2[J2 * 2 + K2, r_in]
                    
    for k in range(4):
        for i in range(4):
            K1, K2 = divmod(k, 2)
            I1, I2 = divmod(i, 2)
            row_w = k * 4 + i
            for r_out in range(7):
                for r_in in range(7):
                    r = r_out * 7 + r_in
                    W4[row_w, r] = W2[K1 * 2 + I1, r_out] * W2[K2 * 2 + I2, r_in]
                    
    # Construct 4x4x8 factors
    # Rank 98
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # Block 1 (k=0..3)
    U[:, 0:49] = U4
    # V rows j*8+k for k=0..3
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, 0:49]
    # W rows k*4+i for k=0..3
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, 0:49]
            
    # Block 2 (k=4..7)
    U[:, 49:98] = U4
    # V rows j*8+k for k=4..7
    for j in range(4):
        for k in range(4, 8):
            V[j * 8 + k, 49:98] = V4[j * 4 + (k-4), 0:49]
    # W rows k*4+i for k=4..7
    for k in range(4, 8):
        for i in range(4):
            W[k * 4 + i, 49:98] = W4[(k-4) * 4 + i, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
