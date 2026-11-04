"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x2x8 using Strassen (4 blocks of 2x2x2)
    # A is 2x3, B is 3x8, C is 2x8
    # We use A[0:2, 0:2] and B[0:2, :] for this part
    
    # Strassen coefficients
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
    # M2: (A10+A11)B00 -> C10+1, C11-1
    # M3: A00(B01-B11) -> C01+1, C11+1
    # M4: A11(B10-B00) -> C00+1, C10+1
    # M5: (A00+A01)B11 -> C00-1, C01+1
    # M6: (A10-A00)(B00+B01) -> C11+1
    # M7: (A01-A11)(B10+B11) -> C00+1
    
    strassen_defs = [
        # (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
        # U index: i*3+j (j in 0,1)
        # V index: j*8+k (j in 0,1)
        # W index: k*2+i
        
        # M1
        ([0, 4], [1, 1], [0, 9], [1, 1], [0, 3], [1, 1]), # Block 0 k=0,1 -> V[0], V[9] (1*8+1)
        # Wait, V index for B11 is 1*8 + 1 = 9.
        # W indices for block 0: C00=0, C10=1, C01=2, C11=3.
        
        # M2
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        
        # M3
        ([0], [1], [1, 9], [1, -1], [2, 3], [1, 1]),
        
        # M4
        ([4], [1], [8, 0], [1, -1], [0, 1], [1, 1]),
        
        # M5
        ([0, 1], [1, 1], [9], [1], [0, 2], [-1, 1]),
        
        # M6
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        
        # M7
        ([1, 4], [1, -1], [8, 9], [1, 1], [0], [1]),
    ]
    
    for t in range(4):
        base_r = t * 7
        # V offset for k: 2t, 2t+1
        # V indices in defs are relative to block start?
        # My defs above used absolute V indices for block 0.
        # I need to adjust V indices for each block t.
        # V index j*8+k.
        # For block t, k is 2t or 2t+1.
        # So V index = j*8 + 2t + delta_k.
        
        for s in range(7):
            r = base_r + s
            u_idx, u_val, v_idx, v_val, w_idx, w_val = strassen_defs[s]
            
            for idx, val in zip(u_idx, u_val):
                U[idx][r] = val
                
            for idx, val in zip(v_idx, v_val):
                # idx in defs was for k=0,1.
                # Actual k = 2t + idx % 2? No.
                # In defs, V[0] meant B00 (k=0), V[1] meant B01 (k=1).
                # V[8] meant B10 (k=0), V[9] meant B11 (k=1).
                # So idx % 2 gives the k offset within the block.
                # idx // 8 gives the j index (0 or 1).
                j = idx // 8
                k_offset = idx % 8
                # k_offset is 0 or 1.
                actual_k = 2 * t + k_offset
                actual_v_idx = j * 8 + actual_k
                V[actual_v_idx][r] = val

            for idx, val in zip(w_idx, w_val):
                # idx in defs was 0,1,2,3 for C00, C10, C01, C11.
                # Actual W index: k*2 + i.
                # C00: i=0, k=0 (local) -> k_global = 2t. W[2t*2 + 0] = 4t.
                # C10: i=1, k=0 (local) -> k_global = 2t. W[2t*2 + 1] = 4t+1.
                # C01: i=0, k=1 (local) -> k_global = 2t+1. W[(2t+1)*2 + 0] = 4t+2.
                # C11: i=1, k=1 (local) -> k_global = 2t+1. W[(2t+1)*2 + 1] = 4t+3.
                actual_w_idx = 4 * t + idx
                W[actual_w_idx][r] = val

    # Part 2: 2x1x8 trivial
    # Uses A[:, 2] and B[2, :]
    base_r = 28
    for i in range(2):
        for k in range(8):
            r = base_r + i * 8 + k
            U[i * 3 + 2][r] = 1
            V[2 * 8 + k][r] = 1
            W[k * 2 + i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
