import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct Strassen 2x2x2 decomposition (rank 7)
    # U, V, W are 4x7
    # Indices: 0:00, 1:01, 2:10, 3:11
    # W indices: k*n+i. n=2.
    # 0: C00, 1: C10, 2: C01, 3: C11
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # M2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # M3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1 # C01, C11
    
    # M4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1 # C00, C10
    
    # M5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1 # C00, C01
    
    # M6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # M7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00

    # 2. Construct 4x4x4 decomposition via Kronecker product
    # U4 = U2 (x) U2, shape 16x49
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # 3. Construct 4x4x1 decomposition (rank 16)
    # U1: 16x16, V1: 16x16, W1: 4x16
    # U1[i,j, r] = 1 if r == i*4+j
    # V1[j,k, r] = 1 if r == i*4+j and k==0
    # W1[k,i, r] = 1 if r == i*4+j and k==0
    
    U1 = np.eye(16)
    V1 = np.zeros((16, 16))
    for r in range(16):
        i = r // 4
        j = r % 4
        V1[j, 0] = 1 # B is 4x1, so j is 0..3, k is 0. Index j*1+0 = j.
        # Wait, V shape is m*p = 4*1 = 4.
        # V1 should be 4x16.
        pass
        
    # Let's redo V1 and W1 shapes.
    # m=4, p=1. V is 4x16.
    # V[j, r] = 1 if r corresponds to (i,j).
    V1 = np.zeros((4, 16))
    for r in range(16):
        i = r // 4
        j = r % 4
        V1[j, r] = 1
        
    # W is p*n = 4*4 = 16.
    # W[k*n+i, r]. k=0. Index i.
    W1 = np.zeros((16, 16))
    for r in range(16):
        i = r // 4
        j = r % 4
        W1[i, r] = 1 # k=0, so index 0*4+i = i.

    # 4. Assemble full matrices
    rank = 49 + 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((36, rank)) # m*p = 4*9 = 36
    W = np.zeros((36, rank)) # p*n = 9*4 = 36
    
    # Block 1: 4x4x4 (cols 0-3)
    # U4 is 16x49. Map to U[0:16, 0:49]
    U[0:16, 0:49] = U4
    # V4 is 16x49. Map to V[0:16, 0:49] (cols 0-3 of B)
    V[0:16, 0:49] = V4
    # W4 is 16x49. Map to W[0:16, 0:49] (cols 0-3 of C)
    # W index k*4+i. k=0..3.
    W[0:16, 0:49] = W4
    
    # Block 2: 4x4x4 (cols 4-7)
    U[0:16, 49:98] = U4
    # V cols 4-7. V index j*9+k. j=0..3, k=4..7.
    # V4 is 16x49. We need to place it in V[j*9 + (4+k), r].
    # V4 row j*4+k_local.
    for j in range(4):
        for k_local in range(4):
            k_global = 4 + k_local
            v_idx = j * 9 + k_global
            v4_idx = j * 4 + k_local
            V[v_idx, 49:98] = V4[v4_idx, :]
            
    # W cols 4-7. W index k*4+i. k=4..7.
    # W4 row k_local*4+i.
    for k_local in range(4):
        k_global = 4 + k_local
        w_idx_start = k_global * 4
        w4_idx_start = k_local * 4
        W[w_idx_start:w_idx_start+4, 49:98] = W4[w4_idx_start:w4_idx_start+4, :]

    # Block 3: 4x4x1 (col 8)
    U[0:16, 98:114] = U1
    # V col 8. V index j*9+8.
    for j in range(4):
        v_idx = j * 9 + 8
        V[v_idx, 98:114] = V1[j, :]
        
    # W col 8. W index 8*4+i = 32+i.
    for i in range(4):
        w_idx = 32 + i
        W[w_idx, 98:114] = W1[i, :]

    return {"U": U, "V": V, "W": W}
