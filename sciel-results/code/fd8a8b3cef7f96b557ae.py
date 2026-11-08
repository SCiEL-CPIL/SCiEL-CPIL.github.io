import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    # Strassen decomposition for 2x2x2
    # U1: 4x7, V1: 4x7, W1: 4x7
    # Indices for 2x2 matrix: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00 + A11)(B00 + B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # M2 = (A00 + A11)B10
    U1[0, 1] = 1; U1[3, 1] = 1
    V1[2, 1] = 1
    W1[2, 1] = 1  # C10 += M2
    W1[3, 1] = -1 # C11 -= M2
    
    # M3 = A00(B01 - B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[1, 2] = 1  # C01 += M3
    W1[3, 2] = 1  # C11 += M3
    
    # M4 = A11(B10 - B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1  # C00 += M4
    W1[2, 3] = 1  # C10 += M4
    
    # M5 = (A01 + A11)B11
    U1[1, 4] = 1; U1[3, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1 # C00 -= M5
    W1[1, 4] = 1  # C01 += M5
    
    # M6 = (A00 - A01)(B00 + B01)
    U1[0, 5] = 1; U1[1, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1  # C11 += M6
    
    # M7 = (A10 - A00)(B10 + B11)
    U1[2, 6] = 1; U1[0, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1  # C00 += M7
    W1[2, 6] = 1  # C10 += M7

    # Schoolbook decomposition for 1x2x3
    # U2: 2x6, V2: 6x6, W2: 3x6
    # A indices: 0: (0,0), 1: (0,1)
    # B indices: 0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0), 4: (1,1), 5: (1,2)
    # C indices: 0: (0,0), 1: (0,1), 2: (0,2)
    
    U2 = np.zeros((2, 6), dtype=float)
    V2 = np.zeros((6, 6), dtype=float)
    W2 = np.zeros((3, 6), dtype=float)
    
    col = 0
    for j in range(2):
        for k in range(3):
            U2[j, col] = 1
            V2[j*3 + k, col] = 1
            W2[k, col] = 1
            col += 1

    # Combine
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 1, 2, 3
    
    n, m, p = n1*n2, m1*m2, p1*p2
    rank = 7 * 6
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for c1 in range(7):
        for c2 in range(6):
            r = c1 * 6 + c2
            
            # U: index i = i1 * n2 + i2
            for i1 in range(n1):
                for i2 in range(n2):
                    i = i1 * n2 + i2
                    U[i, r] = U1[i1, c1] * U2[i2, c2]
            
            # V: index j = j1 * m2 + j2
            for j1 in range(m1):
                for j2 in range(m2):
                    j = j1 * m2 + j2
                    V[j, r] = V1[j1, c1] * V2[j2, c2]
            
            # W: index k = k1 * p2 + k2, row in W is k*n + i
            # Wait, W matrix in interface is (p*n, rank).
            # The tensor factor W_r corresponds to the output matrix C.
            # C has indices (i, k) where i is row, k is col.
            # The interface W has rows indexed by k*n + i.
            # So for a given r, W[k*n + i, r] should be the coefficient for C[i,k].
            # In our decomposition, W1 and W2 give coefficients for C1[i1, k1] and C2[i2, k2].
            # The combined C[i, k] gets contribution W1[i1, k1] * W2[i2, k2].
            # Here i = i1 * n2 + i2, k = k1 * p2 + k2.
            for i1 in range(n1):
                for k1 in range(p1):
                    for i2 in range(n2):
                        for k2 in range(p2):
                            i = i1 * n2 + i2
                            k = k1 * p2 + k2
                            w_idx = k * n + i
                            W[w_idx, r] = W1[i1 * p1 + k1, c1] * W2[i2 * p2 + k2, c2]
                            
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
