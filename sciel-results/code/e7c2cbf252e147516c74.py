import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # We decompose 4x4x8 into two 4x4x4 blocks.
    # Block 1: A (4x4) * B_left (4x4) -> C_left (4x4)
    # Block 2: A (4x4) * B_right (4x4) -> C_right (4x4)
    
    # Trivial decomposition for 4x4x4
    # Rank is 4*4*4 = 64
    rank_4 = 64
    U_4 = np.zeros((16, rank_4), dtype=int)
    V_4 = np.zeros((16, rank_4), dtype=int)
    W_4 = np.zeros((16, rank_4), dtype=int)
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                r = i * 16 + j * 4 + k
                # U index: i*m + j = i*4 + j
                U_4[i * 4 + j, r] = 1
                # V index: j*p + k = j*4 + k
                V_4[j * 4 + k, r] = 1
                # W index: k*n + i = k*4 + i
                W_4[k * 4 + i, r] = 1
                
    # Construct full matrices
    total_rank = 2 * rank_4
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((32, total_rank), dtype=int)
    W = np.zeros((32, total_rank), dtype=int)
    
    # Block 1 (cols 0-3 of B, cols 0-3 of C)
    U[:, 0:rank_4] = U_4
    V[0:16, 0:rank_4] = V_4
    W[0:16, 0:rank_4] = W_4
    
    # Block 2 (cols 4-7 of B, cols 4-7 of C)
    # Note: B index for second block is j*8 + (k+4) ?
    # V shape is (m*p, rank) = (4*8, rank) = (32, rank).
    # Indices for V: row j (0..3), col k (0..7).
    # For block 1, k is 0..3. V index j*8 + k.
    # For block 2, k is 4..7. V index j*8 + (k+4) ? No, k is the global column index.
    # In the trivial loop above, V_4 was for 4x4x4, so V_4 index was j*4+k.
    # But in the full V matrix, the index is j*8+k.
    # So for block 1 (k=0..3), V[j*8+k] corresponds to V_4[j*4+k].
    # For block 2 (k=4..7), V[j*8+k] corresponds to V_4[j*4+(k-4)].
    
    # Let's re-verify the mapping.
    # V_4 was constructed for a 4x4x4 multiplication.
    # V_4 row index was j*4 + k_local.
    # In the full V, row index is j*8 + k_global.
    
    # Block 1: k_global = 0..3. k_local = k_global.
    # V[0:16, 0:rank_4] = V_4 is correct if V_4's rows map to V's rows 0..15.
    # V_4 row r_v corresponds to j, k_local.
    # V row r_v_full corresponds to j, k_global.
    # For block 1, k_global = k_local. So V[j*8 + k] = V_4[j*4 + k].
    # This matches V[0:16, ...] = V_4 because V_4 has 16 rows (0..15).
    # Row 0 of V_4 is j=0, k=0. Row 0 of V is j=0, k=0.
    # Row 1 of V_4 is j=0, k=1. Row 1 of V is j=0, k=1.
    # ...
    # Row 15 of V_4 is j=3, k=3. Row 15 of V is j=3, k=3.
    # So V[0:16, 0:rank_4] = V_4 is correct.
    
    # Block 2: k_global = 4..7. k_local = k_global - 4.
    # V row for j, k_global is j*8 + k_global.
    # V_4 row for j, k_local is j*4 + k_local.
    # We need V[j*8 + k_global, rank_4 + r] = V_4[j*4 + k_local, r].
    # This is not a simple slice assignment V[16:32] = V_4 because the row indices in V are interleaved or shifted.
    # V rows 16..31 correspond to j=0..3, k=4..7?
    # Row 16: j=2, k=0? No.
    # V index = j*8 + k.
    # j=0: 0..7
    # j=1: 8..15
    # j=2: 16..23
    # j=3: 24..31
    # So V[0:16] covers j=0,1.
    # V[16:32] covers j=2,3.
    # This is not "Block 1" vs "Block 2" in terms of rows.
    # The "Block" concept applies to the *columns* of B and C, which correspond to the *k* index.
    # The *rows* of V are determined by j and k.
    # So for Block 1 (k=0..3), the non-zero entries in V are at rows j*8 + k where k in 0..3.
    # For Block 2 (k=4..7), the non-zero entries in V are at rows j*8 + k where k in 4..7.
    
    # My previous assignment V[0:16, 0:rank_4] = V_4 is WRONG for the second block logic if I just copy.
    # I need to fill V correctly.
    
    # Let's just build the full U, V, W directly to avoid mapping errors.
    
    U = np.zeros((16, 128), dtype=int)
    V = np.zeros((32, 128), dtype=int)
    W = np.zeros((32, 128), dtype=int)
    
    # Block 1: k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                r = i * 16 + j * 4 + k
                U[i * 4 + j, r] = 1
                V[j * 8 + k, r] = 1
                W[k * 4 + i, r] = 1
                
    # Block 2: k in 4..7
    for i in range(4):
        for j in range(4):
            for k in range(4, 8):
                r = 64 + i * 16 + j * 4 + (k - 4)
                U[i * 4 + j, r] = 1
                V[j * 8 + k, r] = 1
                W[k * 4 + i, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
