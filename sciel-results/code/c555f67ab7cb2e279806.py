import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (a+d)(e+h)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (c+d)e
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3: a(f-h)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4: d(g-e)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5: (a+b)h
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6: (c-a)(e+f)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7: (b-d)(g+h)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = 1

    # Construct 4x4x4 factors (rank 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for k in range(7):
        for l in range(7):
            c = 7 * k + l
            for i1 in range(2):
                for j1 in range(2):
                    bA = 2 * i1 + j1
                    for i2 in range(2):
                        for j2 in range(2):
                            idxA = 4 * (2 * i1 + i2) + (2 * j1 + j2)
                            U4[idxA, c] += U2[bA, k] * U2[i2, l]
                            
                            # V index: j, k_block
                            # j = 2*j1 + j2, k_block = 2*j1 + k1? No.
                            # V is for B. B is 4x4.
                            # B blocks are indexed by (j1, k1).
                            # bB = 2*j1 + k1.
                            # Scalar indices j2, k2.
                            # idxB = 4*j + k = 4*(2*j1+j2) + (2*k1+k2)
                            
                            for k1 in range(2):
                                bB = 2 * j1 + k1
                                for j2 in range(2):
                                    for k2 in range(2):
                                        idxB = 4 * (2 * j1 + j2) + (2 * k1 + k2)
                                        V4[idxB, c] += V2[bB, k] * V2[j2, l]
                                        
                                        # W index: k_block, i_block
                                        # k_block = 2*k1 + k2? No.
                                        # C blocks indexed by (k1, i1).
                                        # bC = 2*k1 + i1.
                                        # Scalar indices k2, i2.
                                        # idxC = 4*k + i = 4*(2*k1+k2) + (2*i1+i2)
                                        
                                        bC = 2 * k1 + i1
                                        idxC = 4 * (2 * k1 + k2) + (2 * i1 + i2)
                                        W4[idxC, c] += W2[bC, k] * W2[k2, l]

    # Construct 4x4x1 factors (rank 16)
    U5 = np.zeros((16, 16))
    V5 = np.zeros((20, 16)) # m*p = 4*5
    W5 = np.zeros((20, 16)) # p*n = 5*4
    
    for c in range(16):
        i = c // 4
        j = c % 4
        U5[i * 4 + j, c] = 1
        # B index for last column (k=4)
        V5[j * 5 + 4, c] = 1
        # C index for last column (k=4)
        W5[4 * 4 + i, c] = 1

    # Combine
    U = np.hstack([U4, U5])
    V = np.hstack([V4, V5])
    W = np.hstack([W4, W5])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
