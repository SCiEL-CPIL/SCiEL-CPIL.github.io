import numpy as np

def construct():
    """
    Returns a rank-98 tensor decomposition for 4x4 * 4x8 matrix multiplication.
    """
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Construct 4x4x4 factors (Rank 49)
    # U4: 16x49, V4: 16x49, W4: 16x49
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for idx in range(16):
        i = idx // 4
        j = idx % 4
        i1, i2 = i // 2, i % 2
        j1, j2 = j // 2, j % 2
        
        for t in range(7):
            for s in range(7):
                c = t*7 + s
                U4[idx, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
                
    for idx in range(16):
        j = idx // 4
        k = idx % 4
        j1, j2 = j // 2, j % 2
        k1, k2 = k // 2, k % 2
        
        for t in range(7):
            for s in range(7):
                c = t*7 + s
                V4[idx, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]

    for idx in range(16):
        k = idx // 4
        i = idx % 4
        k1, k2 = k // 2, k % 2
        i1, i2 = i // 2, i % 2
        
        for t in range(7):
            for s in range(7):
                c = t*7 + s
                W4[idx, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]

    # Construct 4x4x8 factors (Rank 98)
    # U: 16x98
    # V: 32x98
    # W: 32x98
    
    U = np.zeros((16, 98), dtype=float)
    V = np.zeros((32, 98), dtype=float)
    W = np.zeros((32, 98), dtype=float)
    
    # Block 1: B[:, 0:4]
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, 0:49]
            
    # Block 2: B[:, 4:8]
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + (k+4), 49:98] = V4[j*4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4 + i, 49:98] = W4[k*4 + i, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
