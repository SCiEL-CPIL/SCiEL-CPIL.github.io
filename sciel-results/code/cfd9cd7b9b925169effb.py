import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Step 1: Construct Strassen decomposition for 2x2x2
    # Rank 7
    # U1: 4x7, V1: 4x7, W1: 4x7
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # c=0: M1 = (A00+A11)(B00+B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1 # C00, C11
    
    # c=1: M2 = (A10+A11)B00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = 1 # C10
    
    # c=2: M3 = A00(B01-B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[1, 2] = 1 # C01
    
    # c=3: M4 = A11(B10-B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[2, 3] = 1; W1[0, 3] = 1 # C10, C00
    
    # c=4: M5 = (A00+A10)B11
    U1[0, 4] = 1; U1[2, 4] = 1
    V1[3, 4] = 1
    W1[1, 4] = 1; W1[0, 4] = -1 # C01, C00
    
    # c=5: M6 = (A01-A00)(B00+B10)
    U1[1, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[2, 5] = 1
    W1[3, 5] = 1 # C11
    
    # c=6: M7 = (A10-A11)(B01+B11)
    U1[2, 6] = 1; U1[3, 6] = -1
    V1[1, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1 # C00
    
    # Verify U1, V1, W1
    # T1[i,j,k] = sum_c U1[i*2+j, c] V1[j*2+k, c] W1[k*2+i, c]
    # Should be 1 for all i,j,k in 0..1
    
    # Step 2: Construct decomposition for 2x2x4
    # Rank 14
    # U2: 4x14, V2: 8x14, W2: 8x14
    # Indices: U2[i*2+j, c], V2[j*4+k, c], W2[k*2+i, c]
    
    U2 = np.zeros((4, 14))
    V2 = np.zeros((8, 14))
    W2 = np.zeros((8, 14))
    
    # Copy 1: k in 0,1
    U2[:, 0:7] = U1
    # V2 rows j*4+k. For k=0,1:
    # V1 rows j*2+k.
    # Map V1[j*2+k] to V2[j*4+k]
    for j in range(2):
        for k in range(2):
            V2[j*4+k, 0:7] = V1[j*2+k, 0:7]
    # W2 rows k*2+i. For k=0,1:
    # W1 rows k*2+i.
    # Map W1[k*2+i] to W2[k*2+i]
    for k in range(2):
        for i in range(2):
            W2[k*2+i, 0:7] = W1[k*2+i, 0:7]
            
    # Copy 2: k in 2,3
    U2[:, 7:14] = U1
    # V2 rows j*4+k. For k=2,3:
    # Map V1[j*2+(k-2)] to V2[j*4+k]
    for j in range(2):
        for k in range(2, 4):
            V2[j*4+k, 7:14] = V1[j*2+(k-2), 0:7]
    # W2 rows k*2+i. For k=2,3:
    # Map W1[(k-2)*2+i] to W2[k*2+i]
    for k in range(2, 4):
        for i in range(2):
            W2[k*2+i, 7:14] = W1[(k-2)*2+i, 0:7]

    # Step 3: Kronecker Product
    # U: 16x98, V: 32x98, W: 32x98
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    for c1 in range(7):
        for c2 in range(14):
            c = c1 * 14 + c2
            # U
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            idx_u = (i1*2 + i0)*4 + (j1*2 + j0)
                            U[idx_u, c] = U1[i1*2 + j1, c1] * U2[i0*2 + j0, c2]
            # V
            for j1 in range(2):
                for k1 in range(2):
                    for j0 in range(2):
                        for k2 in range(4):
                            idx_v = (j1*2 + j0)*8 + (k1*4 + k2)
                            V[idx_v, c] = V1[j1*2 + k1, c1] * V2[j0*4 + k2, c2]
            # W
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(4):
                        for i0 in range(2):
                            idx_w = (k1*4 + k2)*4 + (i1*2 + i0)
                            W[idx_w, c] = W1[k1*2 + i1, c1] * W2[k2*2 + i0, c2]
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
