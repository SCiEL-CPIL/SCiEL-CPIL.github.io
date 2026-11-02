import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Step 1: Define 2x2x2 Strassen decomposition
    # Rank 7
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11+A22)(B11+B22)
    # A11 idx 0, A22 idx 3
    # B11 idx 0, B22 idx 3
    # C11 idx 0, C22 idx 3, C12 idx 1, C21 idx 2
    # Wait, W index is k*2+i.
    # C11: k=0, i=0 -> 0
    # C12: k=0, i=1 -> 1
    # C21: k=1, i=0 -> 2
    # C22: k=1, i=1 -> 3
    
    # Col 0: M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1; W2[1, 0] = -1; W2[2, 0] = -1 # C11=M1, C22=M1, C12=-M1? No.
    # Let's re-verify Strassen formulas for C.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So W entries for each M:
    # M1: C11(+1), C22(+1) -> W[0]=1, W[3]=1
    # M2: C21(+1), C22(-1) -> W[2]=1, W[3]=-1
    # M3: C12(+1), C22(+1) -> W[1]=1, W[3]=1
    # M4: C11(+1), C21(+1) -> W[0]=1, W[2]=1
    # M5: C11(-1), C12(+1) -> W[0]=-1, W[1]=1
    # M6: C22(+1) -> W[3]=1
    # M7: C11(+1) -> W[0]=1
    
    # Let's populate U, V, W for 2x2x2
    
    # M1: (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A21+A22)B11
    # A21 idx 2, A22 idx 3. B11 idx 0.
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3: A11(B12-B22)
    # A11 idx 0. B12 idx 1, B22 idx 3.
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4: A22(B21-B11)
    # A22 idx 3. B21 idx 2, B11 idx 0.
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5: (A11+A12)B22
    # A11 idx 0, A12 idx 1. B22 idx 3.
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6: (A21-A11)(B11+B12)
    # A21 idx 2, A11 idx 0. B11 idx 0, B12 idx 1.
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22)
    # A12 idx 1, A22 idx 3. B21 idx 2, B22 idx 3.
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Step 2: Compose to 4x4x4 (Rank 49)
    # This is a tensor product of the 2x2x2 Strassen with itself.
    # U4 = U2 kron U2 ? No, it's a specific composition.
    # The 4x4 matrix A is split into 4 blocks of 2x2.
    # The "scalar" multiplications in the outer Strassen are 2x2x2 multiplications.
    # So we replace each "1" in U2 with the corresponding U2 block?
    # Actually, the rank is 7*7=49.
    # The columns of U4 are indexed by (c1, c2) where c1,c2 in 0..6.
    # U4[i*4+j, c1*7+c2] = U2[i*2+j1, c1] * U2[j1*2+j2, c2]?
    # This is getting complicated to derive by hand.
    # Easier: Just use the known fact that R(4x4x4) <= 49 and generate the matrices via Kronecker products of the 2x2x2 decomposition?
    # If we use the Kronecker product of the 2x2x2 Strassen decomposition with itself, we get a decomposition of 4x4x4.
    # Rank is 7*7=49.
    # U4 = U2 kron U2
    # V4 = V2 kron V2
    # W4 = W2 kron W2
    # Let's verify if this works.
    # (U2 kron U2) (V2 kron V2) (W2 kron W2) = (U2 V2 W2) kron (U2 V2 W2) = T2 kron T2 = T4.
    # Yes, the tensor product of two decompositions is a decomposition of the product tensor.
    # T2 is the 2x2x2 multiplication tensor. T2 kron T2 is the 4x4x4 multiplication tensor.
    # So U4 = np.kron(U2, U2), etc.
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Step 3: Extend to 4x4x8
    # Rank 98
    U = np.zeros((16, 98), dtype=int)
    V = np.zeros((32, 98), dtype=int)
    W = np.zeros((32, 98), dtype=int)
    
    # First 49 columns: k=0..3
    U[:, 0:49] = U4
    # V4 is 16x49. Rows j*4+k for k in 0..3.
    # V is 32x98. Rows j*8+k for k in 0..7.
    # For k in 0..3, row j*8+k corresponds to V4 row j*4+k.
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, :]
            
    # W4 is 16x49. Rows k*4+i for k in 0..3.
    # W is 32x98. Rows k*4+i for k in 0..7.
    # For k in 0..3, row k*4+i corresponds to W4 row k*4+i.
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, :]
            
    # Next 49 columns: k=4..7
    U[:, 49:98] = U4
    # V: k in 4..7. Row j*8+k.
    # Corresponds to V4 row j*4+(k-4).
    for j in range(4):
        for k in range(4, 8):
            V[j*8+k, 49:98] = V4[j*4+(k-4), :]
            
    # W: k in 4..7. Row k*4+i.
    # Corresponds to W4 row (k-4)*4+i.
    for k in range(4, 8):
        for i in range(4):
            W[k*4+i, 49:98] = W4[(k-4)*4+i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
