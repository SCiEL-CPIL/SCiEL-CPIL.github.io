"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen decomposition for 2x2
    # U_2, V_2, W_2
    # Inputs are 2x2 matrices, flattened row-major.
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    # U2[0,0]=1, U2[3,0]=1
    # V2[0,0]=1, V2[3,0]=1
    # W2[0,0]=1 (for C00), W2[3,0]=1 (for C11)? No, W maps to C.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's populate based on the formula:
    # c=0: M1. A:00,11. B:00,11. C:00(1), 11(1) -- Wait, M1 contributes to C00 and C11?
    # Formula:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U2 (A part):
    U2[0, 0] = 1; U2[3, 0] = 1 # M1
    U2[2, 1] = 1; U2[3, 1] = 1 # M2
    U2[0, 2] = 1 # M3
    U2[3, 3] = 1 # M4
    U2[0, 4] = 1; U2[1, 4] = 1 # M5
    U2[2, 5] = 1; U2[0, 5] = -1 # M6
    U2[1, 6] = 1; U2[3, 6] = -1 # M7
    
    # V2 (B part):
    V2[0, 0] = 1; V2[3, 0] = 1 # M1
    V2[0, 1] = 1 # M2
    V2[1, 2] = 1; V2[3, 2] = -1 # M3
    V2[2, 3] = 1; V2[0, 3] = -1 # M4
    V2[3, 4] = 1 # M5
    V2[0, 5] = 1; V2[1, 5] = 1 # M6
    V2[2, 6] = 1; V2[3, 6] = 1 # M7
    
    # W2 (C part):
    # C00 (idx 0): M1(1) + M4(1) - M5(1) + M7(1)
    W2[0, 0] = 1
    W2[0, 3] = 1
    W2[0, 4] = -1
    W2[0, 6] = 1
    
    # C01 (idx 1): M3(1) + M5(1)
    W2[1, 2] = 1
    W2[1, 4] = 1
    
    # C10 (idx 2): M2(1) + M4(1)
    W2[2, 1] = 1
    W2[2, 3] = 1
    
    # C11 (idx 3): M1(1) - M2(1) + M3(1) + M6(1)
    W2[3, 0] = 1
    W2[3, 1] = -1
    W2[3, 2] = 1
    W2[3, 5] = 1

    # 2. Define Schoolbook decomposition for 3x3
    # Rank 27
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = i * 3 * 3 + j * 3 + k
                U3[i * 3 + j, c] = 1
                V3[j * 3 + k, c] = 1
                W3[k * 3 + i, c] = 1

    # 3. Combine using Kronecker Product Logic
    # Dimensions: 6 = 2 * 3
    # Index mapping: idx = 3 * idx_2 + idx_3
    # Rank mapping: c = 27 * c_2 + c_3
    
    U = np.zeros((36, 189))
    V = np.zeros((36, 189))
    W = np.zeros((36, 189))
    
    for c2 in range(7):
        for c3 in range(27):
            c = 27 * c2 + c3
            for i2 in range(2):
                for i3 in range(3):
                    i = 3 * i2 + i3
                    val_u = U2[i2, c2] * U3[i3, c3]
                    if val_u != 0:
                        U[i, c] = val_u
                        
            for j2 in range(2):
                for j3 in range(3):
                    j = 3 * j2 + j3
                    val_v = V2[j2, c2] * V3[j3, c3]
                    if val_v != 0:
                        V[j, c] = val_v
                        
            for k2 in range(2):
                for k3 in range(3):
                    k = 3 * k2 + k3
                    val_w = W2[k2, c2] * W3[k3, c3]
                    if val_w != 0:
                        W[k, c] = val_w

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
