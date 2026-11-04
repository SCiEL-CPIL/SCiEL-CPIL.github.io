import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct 2x2 Strassen decomposition (Rank 7)
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = -1
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = 1
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[3, 6] = -1

    # 2. Construct 4x4 Strassen decomposition (Rank 49) via Kronecker Product of 2x2s
    # U44[i*4+j, r1*7+r2] = U2[i*2+i2, r1] * U2[j*2+j2, r2]
    # Indices for 4x4: i, j in 0..3.
    # i = 2*i2 + i1, j = 2*j2 + j1.
    # U2 index: i*2+j -> (2*i2+i1)*2 + (2*j2+j1) = 4*i2 + 2*i1 + 2*j2 + j1.
    # This is getting messy to map directly. 
    # Easier: Iterate over all 4x4 indices and 49 ranks.
    
    U44 = np.zeros((16, 49), dtype=float)
    V44 = np.zeros((16, 49), dtype=float)
    W44 = np.zeros((16, 49), dtype=float)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U44 rows: i*4+j
            for i in range(4):
                for j in range(4):
                    # Decompose i, j into 2-bit parts
                    i2 = i // 2
                    i1 = i % 2
                    j2 = j // 2
                    j1 = j % 2
                    
                    # U2 indices
                    u_idx1 = i2 * 2 + i1
                    u_idx2 = j2 * 2 + j1
                    
                    val = U2[u_idx1, r1] * U2[u_idx2, r2]
                    U44[i * 4 + j, r] = val
                    
            # V44 rows: j*4+k
            for j in range(4):
                for k in range(4):
                    j2 = j // 2
                    j1 = j % 2
                    k2 = k // 2
                    k1 = k % 2
                    
                    v_idx1 = j2 * 2 + j1
                    v_idx2 = k2 * 2 + k1
                    
                    val = V2[v_idx1, r1] * V2[v_idx2, r2]
                    V44[j * 4 + k, r] = val
                    
            # W44 rows: k*4+i
            for k in range(4):
                for i in range(4):
                    k2 = k // 2
                    k1 = k % 2
                    i2 = i // 2
                    i1 = i % 2
                    
                    w_idx1 = k2 * 2 + k1
                    w_idx2 = i2 * 2 + i1
                    
                    val = W2[w_idx1, r1] * W2[w_idx2, r2]
                    W44[k * 4 + i, r] = val

    # 3. Construct 1x1 Trivial decomposition (Rank 1)
    U11 = np.array([[1.0]])
    V11 = np.array([[1.0]])
    W11 = np.array([[1.0]])

    # 4. Combine 4x4 and 1x1 into 5x5
    # Rank 50
    U55 = np.zeros((25, 50), dtype=float)
    V55 = np.zeros((25, 50), dtype=float)
    W55 = np.zeros((25, 50), dtype=float)
    
    # Map 4x4 part (indices 0-3) to 5x5 (indices 0-3)
    for i in range(4):
        for j in range(4):
            row_u = i * 5 + j
            row_v = j * 5 + i # Wait, V index is j*5+k. Here k is the other dim.
            # In 4x4, V index is j*4+k.
            # We need to map 4x4 V rows to 5x5 V rows.
            # 4x4 V row j*4+k corresponds to 5x5 V row j*5+k if k<4.
            pass
            
    # Let's do it systematically.
    # U55[i*5+j, r] for i,j < 4 comes from U44[i*4+j, r]
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U55[i * 5 + j, r] = U44[i * 4 + j, r]
                
    # V55[j*5+k, r] for j,k < 4 comes from V44[j*4+k, r]
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V55[j * 5 + k, r] = V44[j * 4 + k, r]
                
    # W55[k*5+i, r] for k,i < 4 comes from W44[k*4+i, r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W55[k * 5 + i, r] = W44[k * 4 + i, r]
                
    # Map 1x1 part (index 4) to 5x5 (index 4)
    # Rank index 49
    U55[4 * 5 + 4, 49] = 1.0
    V55[4 * 5 + 4, 49] = 1.0
    W55[4 * 5 + 4, 49] = 1.0
    
    # Convert to lists for return
    U = U55.tolist()
    V = V55.tolist()
    W = W55.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
