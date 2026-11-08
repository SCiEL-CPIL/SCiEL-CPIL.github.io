import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    # 2x2x2 Strassen decomposition
    # U, V, W are 4x7
    # Indices: i,j in 0..1. idx = i*2+j
    # U[i*2+j, r]
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0], # i=0, j=0
        [0, 1, 0, 0, 0, 1, -1],# i=0, j=1
        [0, 1, 0, 1, 0, -1, 0],# i=1, j=0
        [1, 0, -1, 1, 1, 0, 1] # i=1, j=1
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 1, 0, 1, 0, 1],  # j=0, k=1
        [0, 0, 0, 1, 0, 1, 1],  # j=1, k=0
        [1, -1, -1, 0, 1, 0, 1] # j=1, k=1
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0
        [0, 0, 1, 0, 1, 0, 0],  # k=0, i=1
        [0, 1, 0, 0, 0, 1, 0],  # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1
    ], dtype=float)
    
    # Total rank: 49 (from 4x4x4) + 16 (from 4x4x1) = 65
    rank = 65
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: 4x4x4 block (first 49 columns)
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            
                            # U index: i*m + j
                            u_val = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
                            U[i*m+j, r] = u_val
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    
                                    # V index: j*p + k
                                    v_val = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
                                    V[j*p+k, r] = v_val
                                    
                                    # W index: k*n + i
                                    w_val = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
                                    W[k*n+i, r] = w_val
    
    # Part 2: 4x4x1 block (next 16 columns)
    # This handles k=4
    col_offset = 49
    for i in range(4):
        for j in range(4):
            col = col_offset + i*4 + j
            
            # U index: i*m + j
            U[i*m+j, col] = 1
            
            # V index: j*p + 4
            V[j*p+4, col] = 1
            
            # W index: 4*n + i
            W[4*n+i, col] = 1
            
    return {"U": U, "V": V, "W": W}
