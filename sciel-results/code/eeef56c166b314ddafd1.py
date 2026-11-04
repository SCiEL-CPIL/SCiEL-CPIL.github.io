import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    # r in 0..6
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11)(B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # C00 += M1, C11 += M1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # C10 += M2, C11 -= M2
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M3 = A00(B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    # C01 += M3, C11 += M3
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    # C00 += M4, C10 += M4
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M5 = (A00 + A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # C00 -= M5, C01 += M5
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # C11 += M6
    W2[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    # C00 += M7
    W2[0, 6] = 1
    
    # Construct 4x4 decomposition
    # Rank = 7 * 7 = 49
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Fill U
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    i = i1 * 2 + i0
                    j = j1 * 2 + j0
                    row_u = i * 4 + j
                    u_outer = U2[i1 * 2 + j1]
                    u_inner = U2[i0 * 2 + j0]
                    for r1 in range(7):
                        for r0 in range(7):
                            col = r1 * 7 + r0
                            U[row_u, col] = u_outer[r1] * u_inner[r0]
                            
    # Fill V
    for j1 in range(2):
        for j0 in range(2):
            for k1 in range(2):
                for k0 in range(2):
                    j = j1 * 2 + j0
                    k = k1 * 2 + k0
                    row_v = j * 4 + k
                    v_outer = V2[j1 * 2 + k1]
                    v_inner = V2[j0 * 2 + k0]
                    for r1 in range(7):
                        for r0 in range(7):
                            col = r1 * 7 + r0
                            V[row_v, col] = v_outer[r1] * v_inner[r0]
                            
    # Fill W
    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    k = k1 * 2 + k0
                    i = i1 * 2 + i0
                    row_w = k * 4 + i
                    w_outer = W2[k1 * 2 + i1]
                    w_inner = W2[k0 * 2 + i0]
                    for r1 in range(7):
                        for r0 in range(7):
                            col = r1 * 7 + r0
                            W[row_w, col] = w_outer[r1] * w_inner[r0]
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
