import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U2[0, 0] = 1 # A11
    U2[3, 0] = 1 # A22
    V2[0, 0] = 1 # B11
    V2[3, 0] = 1 # B22
    W2[0, 0] = 1 # C11
    W2[3, 0] = 1 # C22
    
    # M2 = (A21+A22)B11
    U2[2, 1] = 1 # A21
    U2[3, 1] = 1 # A22
    V2[0, 1] = 1 # B11
    W2[2, 1] = 1 # C21
    W2[3, 1] = 1 # C22
    
    # M3 = A11(B12-B22)
    U2[0, 2] = 1 # A11
    V2[1, 2] = 1 # B12
    V2[3, 2] = -1 # B22
    W2[1, 2] = 1 # C12
    W2[3, 2] = 1 # C22
    
    # M4 = A22(B21-B11)
    U2[3, 3] = 1 # A22
    V2[2, 3] = 1 # B21
    V2[0, 3] = -1 # B11
    W2[2, 3] = 1 # C21
    W2[3, 3] = 1 # C22
    
    # M5 = (A11+A12)B22
    U2[0, 4] = 1 # A11
    U2[1, 4] = 1 # A12
    V2[3, 4] = 1 # B22
    W2[0, 4] = -1 # C11
    W2[3, 4] = 1 # C22
    
    # M6 = (A21-A11)(B11+B12)
    U2[2, 5] = 1 # A21
    U2[0, 5] = -1 # A11
    V2[0, 5] = 1 # B11
    V2[1, 5] = 1 # B12
    W2[3, 5] = 1 # C22
    
    # M7 = (A12-A22)(B21+B22)
    U2[1, 6] = 1 # A12
    U2[3, 6] = -1 # A22
    V2[2, 6] = 1 # B21
    V2[3, 6] = 1 # B22
    W2[0, 6] = 1 # C11
    W2[3, 6] = 1 # C22
    
    # Verify 2x2x2
    # C = U * A * V * B * W? No.
    # T = sum U[r] V[r] W[r]
    # Check:
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = [[19, 22], [43, 50]]
    
    # Construct 4x4x4 factors via Kronecker
    # U4 = U2 (x) U2
    # V4 = V2 (x) V2
    # W4 = W2 (x) W2
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Rank is 49
    rank = 49
    
    # Construct 4x4x8 factors
    # U is same as U4 (16x49)
    U = U4.copy()
    
    # V is 32x49
    # V[j*8+k, r]
    V = np.zeros((32, 49))
    for j in range(4):
        for k in range(8):
            k_idx = k % 4
            V[j*8 + k, :] = V4[j*4 + k_idx, :]
            
    # W is 32x49
    # W[k*4+i, r]
    W = np.zeros((32, 49))
    for k in range(8):
        for i in range(4):
            k_idx = k % 4
            W[k*4 + i, :] = W4[k_idx*4 + i, :]
            
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
