import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen decomposition for 2x2x2
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # 2. Compute Kronecker product for 4x4x4
    # U4 = U2 ⊗ U2, etc.
    # Shapes: (4*4, 7*7) = (16, 49)
    U2_np = np.array(U2)
    V2_np = np.array(V2)
    W2_np = np.array(W2)
    
    U4 = np.kron(U2_np, U2_np)
    V4 = np.kron(V2_np, V2_np)
    W4 = np.kron(W2_np, W2_np)
    
    # 3. Define Schoolbook decomposition for 4x4x1
    # Rank 16
    # U_1: 16x16, V_1: 4x16, W_1: 4x16
    U1 = np.zeros((16, 16))
    V1 = np.zeros((4, 16))
    W1 = np.zeros((4, 16))
    for c in range(16):
        i = c // 4
        j = c % 4
        U1[i*4 + j, c] = 1
        V1[j, c] = 1
        W1[i, c] = 1
        
    # 4. Assemble global matrices
    # Total rank = 49 + 49 + 16 = 114
    rank = 114
    U_global = np.zeros((16, rank))
    V_global = np.zeros((36, rank)) # m*p = 4*9 = 36
    W_global = np.zeros((36, rank)) # p*n = 9*4 = 36
    
    # Part 1: 4x4x4 for B cols 0-3
    # U is same
    U_global[:, 0:49] = U4
    
    # V mapping: local (j, k_l) -> global (j, k_l)
    # V4 index: j*4 + k_l
    # V_global index: j*9 + k_l
    for j in range(4):
        for k_l in range(4):
            V_global[j*9 + k_l, 0:49] = V4[j*4 + k_l, :]
            
    # W mapping: local (k_l, i) -> global (k_l, i)
    # W4 index: k_l*4 + i
    # W_global index: k_l*4 + i
    W_global[0:16, 0:49] = W4
    
    # Part 2: 4x4x4 for B cols 4-7
    U_global[:, 49:98] = U4
    
    # V mapping: local (j, k_l) -> global (j, k_l + 4)
    for j in range(4):
        for k_l in range(4):
            V_global[j*9 + (k_l + 4), 49:98] = V4[j*4 + k_l, :]
            
    # W mapping: local (k_l, i) -> global (k_l + 4, i)
    # W_global index: (k_l+4)*4 + i
    for k_l in range(4):
        for i in range(4):
            W_global[(k_l+4)*4 + i, 49:98] = W4[k_l*4 + i, :]
            
    # Part 3: 4x4x1 for B col 8
    U_global[:, 98:114] = U1
    
    # V mapping: local (j) -> global (j, 8)
    # V1 index: j
    # V_global index: j*9 + 8
    for j in range(4):
        V_global[j*9 + 8, 98:114] = V1[j, :]
        
    # W mapping: local (i) -> global (8, i)
    # W1 index: i
    # W_global index: 8*4 + i = 32 + i
    for i in range(4):
        W_global[32 + i, 98:114] = W1[i, :]
        
    return {
        "U": U_global.tolist(),
        "V": V_global.tolist(),
        "W": W_global.tolist()
    }
