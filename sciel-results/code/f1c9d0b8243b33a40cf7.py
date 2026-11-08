import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1
    U2[0, 0] = 1; U2[2, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[1, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Kronecker products
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # Permutations
    # U4: target i4*4+j4, kron (i_out*2+j_out)*4 + (i_in*2+j_in)
    perm_U = np.zeros(16, dtype=int)
    for i_out in range(2):
        for i_in in range(2):
            for j_out in range(2):
                for j_in in range(2):
                    i4 = i_out * 2 + i_in
                    j4 = j_out * 2 + j_in
                    target = i4 * 4 + j4
                    kron = (i_out * 2 + j_out) * 4 + (i_in * 2 + j_in)
                    perm_U[target] = kron
    U4 = U4_kron[perm_U]
    
    # V4: target j4*4+k4, kron (j_out*2+k_out)*4 + (j_in*2+k_in)
    perm_V = np.zeros(16, dtype=int)
    for j_out in range(2):
        for j_in in range(2):
            for k_out in range(2):
                for k_in in range(2):
                    j4 = j_out * 2 + j_in
                    k4 = k_out * 2 + k_in
                    target = j4 * 4 + k4
                    kron = (j_out * 2 + k_out) * 4 + (j_in * 2 + k_in)
                    perm_V[target] = kron
    V4 = V4_kron[perm_V]
    
    # W4: target k4*4+i4, kron (k_out*2+i_out)*4 + (k_in*2+i_in)
    perm_W = np.zeros(16, dtype=int)
    for k_out in range(2):
        for k_in in range(2):
            for i_out in range(2):
                for i_in in range(2):
                    k4 = k_out * 2 + k_in
                    i4 = i_out * 2 + i_in
                    target = k4 * 4 + i4
                    kron = (k_out * 2 + i_out) * 4 + (k_in * 2 + i_in)
                    perm_W[target] = kron
    W4 = W4_kron[perm_W]
    
    # Construct 4x4x8
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # First half (B1, C1)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Second half (B2, C2)
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
