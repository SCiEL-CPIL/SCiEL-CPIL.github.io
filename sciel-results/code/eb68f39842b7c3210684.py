import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen for 2x2
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2
    U2[2, 1] = 1; U2[3, 1] = 1 # A10, A11
    V2[0, 1] = 1 # B00
    W2[1, 1] = 1; W2[3, 1] = 1 # C10, C11 (k=0,i=1 -> idx 1; k=1,i=1 -> idx 3)
    
    # r=2: M3
    U2[0, 2] = 1 # A00
    V2[1, 2] = 1; V2[3, 2] = -1 # B01, B11
    W2[2, 2] = 1; W2[3, 2] = 1 # C01, C11 (k=1,i=0 -> idx 2; k=1,i=1 -> idx 3)
    
    # r=3: M4
    U2[3, 3] = 1 # A11
    V2[2, 3] = 1; V2[0, 3] = -1 # B10, B00
    W2[1, 3] = 1; W2[0, 3] = 1 # C10, C00 (k=0,i=1 -> idx 1; k=0,i=0 -> idx 0)
    
    # r=4: M5
    U2[0, 4] = 1; U2[1, 4] = 1 # A00, A01
    V2[3, 4] = 1 # B11
    W2[3, 4] = 1; W2[0, 4] = -1 # C11, C00
    
    # r=5: M6
    U2[2, 5] = 1; U2[0, 5] = -1 # A10, A00
    V2[0, 5] = 1; V2[1, 5] = 1 # B00, B01
    W2[3, 5] = 1 # C11
    
    # r=6: M7
    U2[1, 6] = 1; U2[3, 6] = -1 # A01, A11
    V2[2, 6] = 1; V2[3, 6] = 1 # B10, B11
    W2[0, 6] = 1; W2[1, 6] = -1 # C00, C10

    # Schoolbook for 3x3
    n3, m3, p3 = 3, 3, 3
    rank3 = n3 * m3 * p3
    U3 = np.zeros((n3 * m3, rank3), dtype=int)
    V3 = np.zeros((m3 * p3, rank3), dtype=int)
    W3 = np.zeros((p3 * n3, rank3), dtype=int)
    
    col = 0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                U3[i * m3 + j, col] = 1
                V3[j * p3 + k, col] = 1
                W3[k * n3 + i, col] = 1
                col += 1

    # Kronecker Product
    U = np.kron(U2, U3)
    V = np.kron(V2, V3)
    W = np.kron(W2, W3)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
