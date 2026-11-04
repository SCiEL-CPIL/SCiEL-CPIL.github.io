import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U_S, V_S, W_S are 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_S = np.zeros((4, 7))
    V_S = np.zeros((4, 7))
    W_S = np.zeros((4, 7))
    
    # r=0: M1 = (a00+a11)(b00+b11)
    U_S[0, 0] = 1; U_S[3, 0] = 1
    V_S[0, 0] = 1; V_S[3, 0] = 1
    W_S[0, 0] = 1; W_S[3, 0] = 1
    
    # r=1: M2 = (a10+a11)b00
    U_S[2, 1] = 1; U_S[3, 1] = 1
    V_S[0, 1] = 1
    W_S[2, 1] = 1; W_S[3, 1] = 1
    
    # r=2: M3 = a00(b01-b11)
    U_S[0, 2] = 1
    V_S[1, 2] = 1; V_S[3, 2] = -1
    W_S[1, 2] = 1; W_S[3, 2] = 1
    
    # r=3: M4 = a11(b10-b00)
    U_S[3, 3] = 1
    V_S[2, 3] = 1; V_S[0, 3] = -1
    W_S[0, 3] = 1; W_S[2, 3] = 1
    
    # r=4: M5 = (a00+a01)b11
    U_S[0, 4] = 1; U_S[1, 4] = 1
    V_S[3, 4] = 1
    W_S[0, 4] = -1; W_S[1, 4] = 1
    
    # r=5: M6 = (a10-a00)(b00+b01)
    U_S[2, 5] = 1; U_S[0, 5] = -1
    V_S[0, 5] = 1; V_S[1, 5] = 1
    W_S[3, 5] = 1
    
    # r=6: M7 = (a01-a11)(b10+b11)
    U_S[1, 6] = 1; U_S[3, 6] = -1
    V_S[2, 6] = 1; V_S[3, 6] = 1
    W_S[0, 6] = 1
    
    # Construct 4x4x4 decomposition (rank 49)
    # U_4 is 16x49, V_4 is 16x49, W_4 is 16x49
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U_4[i * 4 + j, r] = U_S[i1 * 2 + j1, r1] * U_S[i2 * 2 + j2, r2]
                            
                            V_4[j * 4 + (i1*2+i2), r] # Wait, V index is j*4+k.
                            # Let's re-evaluate V and W indices carefully.
                            # V_4 index: j * 4 + k.
                            # j = j1 * 2 + j2? No.
                            # In 4x4x4, indices are i,j,k in 0..3.
                            # i = i1*2 + i2
                            # j = j1*2 + j2
                            # k = k1*2 + k2
                            # U_4[i*4+j, r] = U_S[i1*2+j1, r1] * U_S[i2*2+j2, r2]
                            # V_4[j*4+k, r] = V_S[j1*2+k1, r1] * V_S[j2*2+k2, r2]
                            # W_4[k*4+i, r] = W_S[k1*2+i1, r1] * W_S[k2*2+i2, r2]
                            
                            # My loop above only iterated i1,j1,i2,j2. I need k1,k2 too.
                            # But U, V, W are separate.
                            # I can fill them in separate loops or one big loop.
                            pass

    # Let's do it properly with separate loops for clarity and correctness
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # Fill U_4
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U_4[i * 4 + j, r] = U_S[i1 * 2 + j1, r1] * U_S[i2 * 2 + j2, r2]
                            
            # Fill V_4
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V_4[j * 4 + k, r] = V_S[j1 * 2 + k1, r1] * V_S[j2 * 2 + k2, r2]
                            
            # Fill W_4
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W_4[k * 4 + i, r] = W_S[k1 * 2 + i1, r1] * W_S[k2 * 2 + i2, r2]

    # Construct 4x4x8 decomposition (rank 98)
    # U: 16 x 98
    # V: 32 x 98
    # W: 32 x 98
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # Block 1 (k in 0..3)
    U[:, 0:49] = U_4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V_4[j * 4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W_4[k * 4 + i, 0:49]
            
    # Block 2 (k in 4..7)
    U[:, 49:98] = U_4
    for j in range(4):
        for k in range(4):
            # k in 4..7 corresponds to k' in 0..3
            V[j * 8 + (k + 4), 49:98] = V_4[j * 4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            # k in 4..7 corresponds to k' in 0..3
            W[(k + 4) * 4 + i, 49:98] = W_4[k * 4 + i, 0:49]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
