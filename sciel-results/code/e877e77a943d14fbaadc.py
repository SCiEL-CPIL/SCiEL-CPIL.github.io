import numpy as np

def construct():
    # Dimensions
    n, m, p = 2, 4, 6
    
    # 1. Strassen decomposition for 2x2x2
    # U1: 4x7, V1: 4x7, W1: 4x7
    U1 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A11
        [0, 0, 0, 0, 1, 0, 1],   # A12
        [0, 1, 0, 0, 0, 1, 0],   # A21
        [1, 1, 0, 1, 0, 0, -1]   # A22
    ], dtype=int)
    
    V1 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B11
        [0, 0, 1, 0, 0, 1, 0],   # B12
        [0, 0, 0, 1, 0, 0, 1],   # B21
        [1, 0, -1, 0, 1, 0, 1]   # B22
    ], dtype=int)
    
    W1 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0]   # C22
    ], dtype=int)
    
    # 2. Schoolbook decomposition for 1x2x3
    # U2: 2x6, V2: 6x6, W2: 3x6
    U2 = np.array([
        [1, 1, 1, 0, 0, 0],  # A1
        [0, 0, 0, 1, 1, 1]   # A2
    ], dtype=int)
    
    V2 = np.eye(6, dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, 0, 0],  # C1
        [0, 1, 0, 0, 1, 0],  # C2
        [0, 0, 1, 0, 0, 1]   # C3
    ], dtype=int)
    
    # 3. Kronecker products
    # U: 8x42
    U_kron = np.kron(U1, U2)
    # V: 24x42
    V_kron = np.kron(V1, V2)
    # W: 12x42
    W_kron = np.kron(W1, W2)
    
    # 4. Permute V and W to standard row-major order
    # V permutation
    perm_V = []
    for r in range(24):
        j = r // 6
        k = r % 6
        j1 = j // 2
        j2 = j % 2
        k1 = k // 3
        k2 = k % 3
        r_kron = (j1 * 2 + k1) * 6 + (j2 * 3 + k2)
        perm_V.append(r_kron)
    V_final = V_kron[perm_V, :]
    
    # W permutation
    perm_W = []
    for r in range(12):
        k = r // 2
        i = r % 2
        k1 = k // 3
        k2 = k % 3
        i1 = i
        r_kron = (k1 * 2 + i1) * 3 + k2
        perm_W.append(r_kron)
    W_final = W_kron[perm_W, :]
    
    # U does not need permutation
    U_final = U_kron
    
    # Convert to list of lists
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
