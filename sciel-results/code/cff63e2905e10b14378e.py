import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # We will build the decomposition in two parts:
    # Part 1: A[0:2, 0:2] * B[0:2, 0:5]
    # Part 2: A[0:2, 2:4] * B[2:4, 0:5]
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to generate 2x2x2 Strassen decomposition
    # Returns U (4x7), V (10x7), W (10x7) for local indices
    # Local i in 0,1; local j in 0,1; local k in 0,1
    def strassen_222():
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        U = np.zeros((4, 7))
        V = np.zeros((10, 7)) # j in 0,1; k in 0,1. Index j*2+k
        W = np.zeros((10, 7)) # k in 0,1; i in 0,1. Index k*2+i
        
        # P1
        U[0, 0] = 1; U[3, 0] = 1 # A11, A22
        V[0, 0] = 1; V[3, 0] = 1 # B11, B22
        W[0, 0] = 1; W[2, 0] = 1 # C11, C22 (k=0,i=0; k=1,i=1) -> Wait, W index is k*n+i. n=2.
        # C11: k=0, i=0 -> idx 0. C22: k=1, i=1 -> idx 3.
        # My W shape is (p*n, R) = (2*2, 7) = (4, 7).
        # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        W[0, 0] = 1 # C11
        W[3, 0] = 1 # C22
        
        # P2
        U[2, 1] = 1; U[3, 1] = 1 # A21, A22
        V[0, 1] = 1 # B11
        W[2, 1] = 1 # C21 (k=1, i=1? No. C21 is i=1, k=0? No. C_{ik}. i=row, k=col.
        # C21 means i=1, k=0. W index = 0*2 + 1 = 1.
        # Let's re-verify W index. W[k*n+i].
        # C11: i=0, k=0 -> 0.
        # C12: i=0, k=1 -> 2.
        # C21: i=1, k=0 -> 1.
        # C22: i=1, k=1 -> 3.
        
        # Re-do W for P1:
        # C11 += P1. W[0,0]=1.
        # C22 += P1. W[3,0]=1.
        
        # Re-do W for P2:
        # C21 += P2. i=1, k=0. W[1,1]=1.
        
        # P3
        U[0, 2] = 1 # A11
        V[1, 2] = 1; V[3, 2] = -1 # B12, B22
        # C12 += P3. i=0, k=1. W[2,2]=1.
        
        # P4
        U[3, 3] = 1 # A22
        V[2, 3] = 1; V[0, 3] = -1 # B21, B11
        # C11 += -P4. W[0,3]=-1.
        # C21 += P4. W[1,3]=1.
        
        # P5
        U[0, 4] = 1; U[1, 4] = 1 # A11, A12
        V[3, 4] = 1 # B22
        # C11 += -P5. W[0,4]=-1.
        # C12 += P5. W[2,4]=1.
        
        # P6
        U[2, 5] = 1; U[0, 5] = -1 # A21, A11
        V[0, 5] = 1; V[1, 5] = 1 # B11, B12
        # C22 += P6. W[3,5]=1.
        
        # P7
        U[1, 6] = 1; U[3, 6] = -1 # A12, A22
        V[2, 6] = 1; V[3, 6] = 1 # B21, B22
        # C11 += P7. W[0,6]=1.
        # C22 += -P7. W[3,6]=-1.
        
        return U, V, W

    # Helper to generate 2x2x1 Schoolbook decomposition
    # Local i in 0,1; local j in 0,1; local k in 0
    def schoolbook_221():
        U = np.zeros((4, 2))
        V = np.zeros((10, 2)) # j*1+0
        W = np.zeros((2, 2))  # k*2+i -> 0*2+i = i
        
        # P1 = A11 B10
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1 # C10
        
        # P2 = A21 B20
        U[3, 1] = 1
        V[2, 1] = 1 # j=1, k=0 -> idx 1*1+0=1
        W[1, 1] = 1 # C20
        
        return U, V, W

    # Build 2x2x5 decomposition
    def build_225():
        # Strassen for cols 0,1
        U1, V1, W1 = strassen_222()
        # Strassen for cols 2,3
        U2, V2, W2 = strassen_222()
        # Schoolbook for col 4
        U3, V3, W3 = schoolbook_221()
        
        U_total = np.hstack([U1, U2, U3])
        
        # V mapping:
        # V1 is for k=0,1. Indices in V1: j*2+k.
        # We need to map to global k=0,1.
        # V2 is for k=2,3.
        # V3 is for k=4.
        
        V_full = np.zeros((10, 16)) # 2*5=10 rows, 16 cols
        # V1 cols 0-6 map to V_full cols 0-6
        V_full[:, 0:7] = V1
        # V2 cols 0-6 map to V_full cols 7-13
        # But V2 indices are j*2+k_local. k_local 0,1 correspond to global k 2,3.
        # So V2[j*2+0] -> V_full[j*2+2]
        # V2[j*2+1] -> V_full[j*2+3]
        for j in range(2):
            for kl in range(2):
                V_full[j*5 + (2+kl), 7+kl*2+j*0] # Wait, V1 shape is (10, 7).
                # V1 row index is j*2+k.
                # We need to place V1 into V_full such that V_full[j*5+k, col] = V1[j*2+k, col].
        
        # Let's do this more carefully.
        V_full = np.zeros((10, 16))
        # Block 1 (k=0,1)
        for j in range(2):
            for k in range(2):
                V_full[j*5+k, 0:7] = V1[j*2+k, 0:7]
        
        # Block 2 (k=2,3)
        for j in range(2):
            for k in range(2):
                V_full[j*5+(2+k), 7:14] = V2[j*2+k, 0:7]
        
        # Block 3 (k=4)
        for j in range(2):
            V_full[j*5+4, 14:16] = V3[j*1+0, 0:2]
            
        # W mapping:
        # W1 is for k=0,1. W1 index k*2+i.
        # W2 is for k=2,3.
        # W3 is for k=4.
        
        W_full = np.zeros((10, 16)) # p*n = 5*2 = 10 rows
        for k in range(2):
            for i in range(2):
                W_full[k*2+i, 0:7] = W1[k*2+i, 0:7]
        
        for k in range(2):
            for i in range(2):
                W_full[(2+k)*2+i, 7:14] = W2[k*2+i, 0:7]
        
        for k in range(1):
            for i in range(2):
                W_full[(4+k)*2+i, 14:16] = W3[k*2+i, 0:2]
                
        return U_total, V_full, W_full

    U_225, V_225, W_225 = build_225()
    
    # Now build 2x4x5
    # Part 1: A cols 0,1. B rows 0,1.
    U1 = np.zeros((8, 16)) # n*m = 2*4=8
    V1 = np.zeros((20, 16)) # m*p = 4*5=20
    W1 = np.zeros((10, 16)) # p*n = 5*2=10
    
    # Map U_225 (4x16) to U1
    # U_225 index i*2+j_local. j_local 0,1 correspond to global j 0,1.
    for i in range(2):
        for j in range(2):
            U1[i*4+j, 0:16] = U_225[i*2+j, 0:16]
            
    # Map V_225 (10x16) to V1
    # V_225 index j_local*5+k. j_local 0,1 correspond to global j 0,1.
    for j in range(2):
        for k in range(5):
            V1[j*5+k, 0:16] = V_225[j*5+k, 0:16]
            
    # Map W_225 (10x16) to W1
    # W_225 index k*2+i. Same as global.
    W1[:, 0:16] = W_225[:, 0:16]
    
    # Part 2: A cols 2,3. B rows 2,3.
    U2 = np.zeros((8, 16))
    V2 = np.zeros((20, 16))
    W2 = np.zeros((10, 16))
    
    # Map U_225 to U2
    # j_local 0,1 correspond to global j 2,3.
    for i in range(2):
        for j in range(2):
            U2[i*4+(2+j), 0:16] = U_225[i*2+j, 0:16]
            
    # Map V_225 to V2
    # j_local 0,1 correspond to global j 2,3.
    for j in range(2):
        for k in range(5):
            V2[(2+j)*5+k, 0:16] = V_225[j*5+k, 0:16]
            
    # Map W_225 to W2
    W2[:, 0:16] = W_225[:, 0:16]
    
    # Combine
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = W1 + W2
    
    return {"U": U, "V": V, "W": W}
