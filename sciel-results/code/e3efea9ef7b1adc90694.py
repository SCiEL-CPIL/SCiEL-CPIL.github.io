import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 decomposition
    # U, V, W are lists of 7 vectors of length 4
    U2 = [
        [1, 0, 0, 1], # M1
        [0, 1, 1, 0], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 1, 0, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    V2 = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, -1, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    W2 = [
        [1, 0, 0, 1], # M1 -> C11, C22
        [0, 0, 1, 0], # M2 -> C21
        [0, 1, 0, 0], # M3 -> C12
        [1, 0, 1, 0], # M4 -> C11, C21
        [-1, 1, 0, 0],# M5 -> C11, C12
        [0, 0, 0, 1], # M6 -> C22
        [1, 0, 0, 0]  # M7 -> C11
    ]
    
    # Construct 4x4x4 decomposition (49 terms)
    U4 = []
    V4 = []
    W4 = []
    for r1 in range(7):
        for r2 in range(7):
            u = [0]*16
            v = [0]*16
            w = [0]*16
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            idx_u = (i1*2 + i2)*4 + (j1*2 + j2) # A is 4x4, row i, col j
                            # Wait, A is n x m = 4x4.
                            # Index in U is i*m + j.
                            # i = i1*2 + i2, j = j1*2 + j2.
                            # U4_idx = (i1*2+i2)*4 + (j1*2+j2)
                            # U2_idx1 = i1*2 + j1
                            # U2_idx2 = i2*2 + j2
                            u_val = U2[r1][i1*2+j1] * U2[r2][i2*2+j2]
                            u[(i1*2+i2)*4 + (j1*2+j2)] = u_val
                            
                            # V is m x p = 4x4.
                            # Index in V is j*p + k.
                            # j = j1*2 + j2, k = k1*2 + k2.
                            # V4_idx = (j1*2+j2)*4 + (k1*2+k2)
                            # V2_idx1 = j1*2 + k1
                            # V2_idx2 = j2*2 + k2
                            for k1 in range(2):
                                for k2 in range(2):
                                    v_val = V2[r1][j1*2+k1] * V2[r2][j2*2+k2]
                                    v[(j1*2+j2)*4 + (k1*2+k2)] = v_val
                                    
                                    # W is p x n = 4x4.
                                    # Index in W is k*n + i.
                                    # k = k1*2 + k2, i = i1*2 + i2.
                                    # W4_idx = (k1*2+k2)*4 + (i1*2+i2)
                                    # W2_idx1 = k1*2 + i1
                                    # W2_idx2 = k2*2 + i2
                                    w_val = W2[r1][k1*2+i1] * W2[r2][k2*2+i2]
                                    w[(k1*2+k2)*4 + (i1*2+i2)] = w_val
            U4.append(u)
            V4.append(v)
            W4.append(w)
            
    # Construct 4x4x8 decomposition (98 terms)
    U = []
    V = []
    W = []
    
    # First half of B (columns 0-3)
    for r in range(49):
        u = U4[r][:]
        v = [0]*32 # m*p = 4*8
        for idx in range(16):
            v[idx] = V4[r][idx]
        # Second half is 0
        w = W4[r][:]
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Second half of B (columns 4-7)
    for r in range(49):
        u = U4[r][:]
        v = [0]*32
        for idx in range(16):
            v[16+idx] = V4[r][idx]
        # W index: k is 4..7.
        # W4 index is k*n + i. k is 0..3.
        # We need to shift k by 4.
        w = [0]*32 # p*n = 8*4
        for idx in range(16):
            k_old = idx // 4
            i = idx % 4
            k_new = k_old + 4
            w[k_new*4 + i] = W4[r][idx]
        U.append(u)
        V.append(v)
        W.append(w)
        
    return {"U": U, "V": V, "W": W}
