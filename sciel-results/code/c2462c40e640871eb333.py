import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen 2x2x2 (Rank 7)
    # U1: (4, 7), V1: (4, 7), W1: (4, 7)
    # Indices:
    # U1[i*2+j, r]
    # V1[j*2+k, r]
    # W1[k*2+i, r]
    
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1 # C00 (k=0,i=0), C11 (k=1,i=1)
    
    # M2 = (A10+A11)B00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = 1 # C10 (k=0,i=1)
    
    # M3 = A00(B01-B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[1, 2] = 1 # C01 (k=1,i=0)
    
    # M4 = A11(B10-B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[2, 3] = 1 # C00, C10
    
    # M5 = (A00+A10)B11
    U1[0, 4] = 1; U1[2, 4] = 1
    V1[3, 4] = 1
    W1[1, 4] = 1; W1[3, 4] = 1 # C01, C11
    
    # M6 = (A11-A00)(B00+B01)
    U1[3, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1; W1[1, 6] = 1 # C00, C01
    
    # 2. Schoolbook 1x2x4 (Rank 8)
    # n2=1, m2=2, p2=4
    # U2: (2, 8), V2: (8, 8), W2: (4, 8)
    # U2[i2*2+j2, r] -> i2=0, so U2[j2, r]
    # V2[j2*4+k2, r]
    # W2[k2*1+i2, r] -> W2[k2, r]
    
    U2 = np.zeros((2, 8), dtype=int)
    V2 = np.zeros((8, 8), dtype=int)
    W2 = np.zeros((4, 8), dtype=int)
    
    for r in range(8):
        # j2 in {0,1}, k2 in {0,1,2,3}
        # We need to cover all (j2, k2) pairs.
        # There are 2*4=8 pairs.
        j2 = r // 4
        k2 = r % 4
        U2[j2, r] = 1
        V2[j2*4 + k2, r] = 1
        W2[k2, r] = 1
        
    # 3. Kronecker Products
    # U = U1 x U2
    # V = V1 x V2
    # W = W1 x W2
    
    U_comp = np.kron(U1, U2)
    V_comp = np.kron(V1, V2)
    W_comp = np.kron(W1, W2)
    
    # 4. Permutations
    # U_comp shape (8, 56). Rows match target directly.
    
    # V_comp shape (32, 56).
    # Target V row index: j*8 + k.
    # j = j1*2 + j2. k = k1*4 + k2.
    # Target idx: (j1*2 + j2)*8 + k1*4 + k2 = 16*j1 + 8*j2 + 4*k1 + k2.
    # Comp V row index: (j1*2 + k1)*8 + (j2*4 + k2) = 16*j1 + 8*k1 + 4*j2 + k2.
    
    V_perm = np.zeros_like(V_comp)
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(4):
                    idx_target = 16*j1 + 8*j2 + 4*k1 + k2
                    idx_comp = 16*j1 + 8*k1 + 4*j2 + k2
                    V_perm[idx_target] = V_comp[idx_comp]
    
    # W_comp shape (16, 56).
    # Target W row index: k*2 + i.
    # k = k1*4 + k2. i = i1.
    # Target idx: (k1*4 + k2)*2 + i1 = 8*k1 + 2*k2 + i1.
    # Comp W row index: (k1*2 + i1)*4 + k2 = 8*k1 + 4*i1 + k2.
    
    W_perm = np.zeros_like(W_comp)
    for k1 in range(2):
        for k2 in range(4):
            for i1 in range(2):
                idx_target = 8*k1 + 2*k2 + i1
                idx_comp = 8*k1 + 4*i1 + k2
                W_perm[idx_target] = W_comp[idx_comp]
                
    # Convert to lists for return
    U_list = U_comp.tolist()
    V_list = V_perm.tolist()
    W_list = W_perm.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
