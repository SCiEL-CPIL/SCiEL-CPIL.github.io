import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Use Kronecker product of Strassen rank-7 decomposition for 4x4 matrices
    # Rank = 7 * 7 = 49
    
    # Strassen 2x2 matrices (U2, V2, W2) of shape (4, 7)
    # Indices: U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1
    
    # M3 = A00(B01-B11) -> C01, C11
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00, C10
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00, C01
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Construct 4x4 matrices
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    # U[i*4+j, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    for i1 in range(2):
        for i2 in range(2):
            i = 2*i1 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2*j1 + j2
                    row_u = i * 4 + j
                    for c1 in range(7):
                        for c2 in range(7):
                            c = 7*c1 + c2
                            U[row_u, c] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
                            
    # V[j*4+k, c1*7+c2] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
    for j1 in range(2):
        for j2 in range(2):
            j = 2*j1 + j2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2*k1 + k2
                    row_v = j * 4 + k
                    for c1 in range(7):
                        for c2 in range(7):
                            c = 7*c1 + c2
                            V[row_v, c] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
                            
    # W[k*4+i, c1*7+c2] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
    for k1 in range(2):
        for k2 in range(2):
            k = 2*k1 + k2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2*i1 + i2
                    row_w = k * 4 + i
                    for c1 in range(7):
                        for c2 in range(7):
                            c = 7*c1 + c2
                            W[row_w, c] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
