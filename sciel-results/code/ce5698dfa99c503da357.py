def construct():
    import numpy as np

    n, m, p = 2, 3, 5
    # We use the block decomposition:
    # 1. 2x2x2 Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. 2x2x2 Strassen for A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 3. 2x2x1 Outer for A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # 4. 2x1x5 Outer for A[0:2, 2:3] * B[2:3, 0:5] -> C[0:2, 0:5]
    
    # Total Rank: 7 + 7 + 4 + 10 = 28
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen_block(rank_offset, j_offset, k_offset):
        # Indices for A (2x2 block)
        a_idx = 0 * m + j_offset
        b_idx = 0 * m + (j_offset + 1)
        c_idx = 1 * m + j_offset
        d_idx = 1 * m + (j_offset + 1)
        
        # Indices for B (2x2 block)
        e_idx = j_offset * p + k_offset
        f_idx = j_offset * p + (k_offset + 1)
        g_idx = (j_offset + 1) * p + k_offset
        h_idx = (j_offset + 1) * p + (k_offset + 1)
        
        # Rank indices
        r0 = rank_offset
        r1 = rank_offset + 1
        r2 = rank_offset + 2
        r3 = rank_offset + 3
        r4 = rank_offset + 4
        r5 = rank_offset + 5
        r6 = rank_offset + 6
        
        # U (A)
        U[a_idx, r0] += 1; U[d_idx, r0] += 1
        U[c_idx, r1] += 1; U[d_idx, r1] += 1
        U[a_idx, r2] += 1
        U[d_idx, r3] += 1
        U[a_idx, r4] += 1; U[b_idx, r4] += 1
        U[c_idx, r5] += 1; U[a_idx, r5] -= 1
        U[b_idx, r6] += 1; U[d_idx, r6] -= 1
        
        # V (B)
        V[e_idx, r0] += 1; V[h_idx, r0] += 1
        V[e_idx, r1] += 1
        V[f_idx, r2] += 1; V[h_idx, r2] -= 1
        V[g_idx, r3] += 1; V[e_idx, r3] -= 1
        V[h_idx, r4] += 1
        V[e_idx, r5] += 1; V[f_idx, r5] += 1
        V[g_idx, r6] += 1; V[h_idx, r6] += 1
        
        # W (C)
        # C_00
        W[k_offset * n + 0, r0] += 1
        W[k_offset * n + 0, r3] += 1
        W[k_offset * n + 0, r4] -= 1
        W[k_offset * n + 0, r6] += 1
        # C_01
        W[(k_offset + 1) * n + 0, r2] += 1
        W[(k_offset + 1) * n + 0, r4] += 1
        # C_10
        W[k_offset * n + 1, r1] += 1
        W[k_offset * n + 1, r3] += 1
        # C_11
        W[(k_offset + 1) * n + 1, r0] += 1
        W[(k_offset + 1) * n + 1, r1] -= 1
        W[(k_offset + 1) * n + 1, r2] += 1
        W[(k_offset + 1) * n + 1, r5] += 1

    # Block 1: j=0,1, k=0,1
    add_strassen_block(0, 0, 0)
    
    # Block 2: j=0,1, k=2,3
    add_strassen_block(7, 0, 2)
    
    # Block 3: j=0,1, k=4 (2x1)
    r0 = 14
    r1 = 15
    r2 = 16
    r3 = 17
    
    U[0 * m + 0, r0] += 1
    U[0 * m + 1, r1] += 1
    U[1 * m + 0, r2] += 1
    U[1 * m + 1, r3] += 1
    
    V[0 * p + 4, r0] += 1
    V[1 * p + 4, r1] += 1
    V[0 * p + 4, r2] += 1
    V[1 * p + 4, r3] += 1
    
    W[4 * n + 0, r0] += 1
    W[4 * n + 0, r1] += 1
    W[4 * n + 1, r2] += 1
    W[4 * n + 1, r3] += 1
    
    # Block 4: j=2, k=0..4 (1x5)
    rank_offset = 18
    for k in range(5):
        r_u0 = rank_offset
        r_u1 = rank_offset + 1
        rank_offset += 2
        
        U[0 * m + 2, r_u0] += 1
        U[1 * m + 2, r_u1] += 1
        
        V[2 * p + k, r_u0] += 1
        V[2 * p + k, r_u1] += 1
        
        W[k * n + 0, r_u0] += 1
        W[k * n + 1, r_u1] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
