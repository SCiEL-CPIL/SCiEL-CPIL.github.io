import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U1, V1, W1 are 4x7 matrices
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # r=0
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # r=1
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = 1; W1[3, 1] = -1
    
    # r=2
    U1[0, 2] = 1
    V1[2, 2] = 1; V1[3, 2] = -1
    W1[1, 2] = 1; W1[3, 2] = 1
    
    # r=3
    U1[3, 3] = 1
    V1[1, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[2, 3] = 1
    
    # r=4
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[1, 4] = 1
    
    # r=5
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # r=6
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1

    # Schoolbook decomposition for 1x2x3
    # U2 is 2x6, V2 is 6x6, W2 is 3x6
    U2 = np.zeros((2, 6), dtype=int)
    V2 = np.zeros((6, 6), dtype=int)
    W2 = np.zeros((3, 6), dtype=int)
    
    for r2 in range(6):
        j2 = r2 // 3
        k2 = r2 % 3
        U2[j2, r2] = 1
        V2[j2 * 3 + k2, r2] = 1
        W2[k2, r2] = 1

    # Construct full decomposition
    R = 7 * 6
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            for i in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        j = j1 * 2 + j2
                        u_val = U1[i * 2 + j1, r1] * U2[j2, r2]
                        if u_val != 0:
                            U[i * 4 + j, r] = u_val
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            j = j1 * 2 + j2
                            k = k1 * 3 + k2
                            v_val = V1[j1 * 2 + k1, r1] * V2[j2 * 3 + k2, r2]
                            if v_val != 0:
                                V[j * 6 + k, r] = v_val
            for k1 in range(2):
                for k2 in range(3):
                    for i in range(2):
                        k = k1 * 3 + k2
                        w_val = W1[k1 * 2 + i, r1] * W2[k2, r2]
                        if w_val != 0:
                            W[k * 2 + i, r] = w_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
