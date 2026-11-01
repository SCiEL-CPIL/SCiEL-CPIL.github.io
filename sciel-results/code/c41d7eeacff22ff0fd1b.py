import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 2, 4, 6
    
    # Rank 42: Kronecker product of 2x2x2 Strassen (rank 7) and 1x2x3 Trivial (rank 6)
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Core 1: 2x2x2 Strassen
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    U1[0, 0] = 1; U1[3, 0] = 1
    U1[2, 1] = 1; U1[3, 1] = 1
    U1[0, 2] = 1
    U1[3, 3] = 1
    U1[0, 4] = 1; U1[1, 4] = 1
    U1[2, 5] = 1; U1[0, 5] = -1
    U1[1, 6] = 1; U1[3, 6] = -1
    
    V1[0, 0] = 1; V1[3, 0] = 1
    V1[0, 1] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    V1[2, 3] = 1; V1[0, 3] = -1
    V1[3, 4] = 1
    V1[0, 5] = 1; V1[1, 5] = 1
    V1[2, 6] = 1; V1[3, 6] = 1
    
    W1[0, 0] = 1; W1[0, 3] = 1; W1[0, 4] = -1; W1[0, 6] = 1
    W1[2, 2] = 1; W1[2, 4] = 1
    W1[1, 1] = 1; W1[1, 3] = 1
    W1[3, 0] = 1; W1[3, 1] = -1; W1[3, 2] = 1; W1[3, 5] = 1
    
    # Core 2: 1x2x3 Trivial
    U2 = np.zeros((2, 6))
    V2 = np.zeros((6, 6))
    W2 = np.zeros((3, 6))
    
    for t in range(6):
        j2 = t // 3
        k2 = t % 3
        U2[j2, t] = 1
        V2[j2 * 3 + k2, t] = 1
        W2[k2, t] = 1
        
    # Kronecker Product Construction
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            for i1 in range(2):
                for j1 in range(2):
                    v1 = U1[i1 * 2 + j1, r1]
                    if v1 != 0:
                        for j2 in range(2):
                            v2 = U2[j2, r2]
                            if v2 != 0:
                                U[i1 * 4 + (j1 * 2 + j2), r] += v1 * v2
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V1[j1 * 2 + k1, r1]
                    if v1 != 0:
                        for j2 in range(2):
                            for k2 in range(3):
                                v2 = V2[j2 * 3 + k2, r2]
                                if v2 != 0:
                                    V[(j1 * 2 + j2) * 6 + (k1 * 3 + k2), r] += v1 * v2
            for i1 in range(2):
                for k1 in range(2):
                    v1 = W1[k1 * 2 + i1, r1]
                    if v1 != 0:
                        for k2 in range(3):
                            v2 = W2[k2, r2]
                            if v2 != 0:
                                W[(k1 * 3 + k2) * 2 + i1, r] += v1 * v2
                                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
