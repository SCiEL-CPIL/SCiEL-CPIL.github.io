"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # A: n x m = 2x3
    # B: m x p = 3x6
    # C: n x p = 2x6
    
    # We use a block decomposition of B into three 3x2 blocks.
    # Each block multiplication is 2x3x2.
    # Rank of 2x3x2 is 11 (7 Strassen terms for 2x2x2 + 4 scalar terms).
    # Total rank = 3 * 11 = 33.
    
    rank = 33
    # U: (n*m) x rank = 6 x 33
    # V: (m*p) x rank = 18 x 33
    # W: (p*n) x rank = 12 x 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 11 terms for a single 2x3x2 block
    # A (2x3): indices 0..5 (i*3+j)
    # B (3x2): indices 0..5 (j*2+k)
    # C (2x2): indices 0..3 (i*2+k)
    
    # Strassen terms for the 2x2x2 part (A[:, 0:2] * B[0:2, :])
    # A_L = [[A00, A01], [A10, A11]] -> flat indices 0,1,3,4
    # B_T = [[B00, B01], [B10, B11]] -> flat indices 0,1,2,3
    
    terms = []
    
    # M1 = (A00 + A11) * (B00 + B11)
    terms.append(({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
    # M2 = (A10 + A11) * B00
    terms.append(({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}))
    # M3 = A00 * (B01 - B11)
    terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
    # M4 = A11 * (B10 - B00)
    terms.append(({4: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
    # M5 = (A00 + A01) * B11
    terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
    # M6 = (A10 - A00) * (B00 + B01)
    terms.append(({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
    # M7 = (A01 - A11) * (B10 + B11)
    terms.append(({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}))
    
    # Scalar terms for the remaining column of A and row of B
    # A[:, 2] is flat indices 2, 5
    # B[2, :] is flat indices 4, 5
    
    # K1 = A02 * B20
    terms.append(({2: 1}, {4: 1}, {0: 1}))
    # K2 = A02 * B21
    terms.append(({2: 1}, {5: 1}, {1: 1}))
    # K3 = A12 * B20
    terms.append(({5: 1}, {4: 1}, {2: 1}))
    # K4 = A12 * B21
    terms.append(({5: 1}, {5: 1}, {3: 1}))
    
    # Fill the global matrices for the 3 blocks
    # Block b corresponds to B columns 2b, 2b+1
    for block in range(3):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = block * 11 + t_idx
            
            # U indices are global A indices (i*m + j)
            for idx, val in u_dict.items():
                U[idx][col] = val
            
            # V indices are global B indices (j*p + k)
            # Local B index l_idx = j_local * 2 + k_local
            # Global k = block * 2 + k_local
            # Global j = j_local
            # Global V index = j_local * p + (block * 2 + k_local)
            for l_idx, val in v_dict.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            # W indices are global C indices (k*n + i)
            # Local C index l_idx = i_local * 2 + k_local
            # Global k = block * 2 + k_local
            # Global i = i_local
            # Global W index = (block * 2 + k_local) * n + i_local
            for l_idx, val in w_dict.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
