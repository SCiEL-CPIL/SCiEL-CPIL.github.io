import numpy as np

def construct():
    N, M, P = 3, 4, 6
    A_DIM = N * M
    B_DIM = M * P
    C_DIM = P * N
    
    # Target rank
    R = 66
    
    U = np.zeros((A_DIM, R))
    V = np.zeros((B_DIM, R))
    W = np.zeros((C_DIM, R))
    
    # Helper to add a block decomposition
    def add_block(u_start, v_start, w_start, u_cols, v_cols, w_cols, u_block, v_block, w_block):
        """
        u_start: start index in global A
        v_start: start index in global B
        w_start: start index in global C
        u_cols: number of cols in u_block
        v_cols: number of cols in v_block
        w_cols: number of cols in w_block
        u_block: local U matrix (u_cols x rank)
        v_block: local V matrix (v_cols x rank)
        w_block: local W matrix (w_cols x rank)
        """
        rank = u_block.shape[1]
        # Map indices
        # This is tricky because the local indices map to global indices non-trivially
        # We will handle mapping in the specific block constructors
        pass

    # 1. Build 2x2x2 Strassen (Rank 7)
    # Local dims: A=2, B=2, C=2
    U_222 = np.zeros((2, 7))
    V_222 = np.zeros((2, 7))
    W_222 = np.zeros((2, 7))
    
    # M1
    U_222[:, 0] = [1, 1]
    V_222[:, 0] = [1, 1]
    W_222[:, 0] = [1, 0, 0, 1] # C00, C01, C10, C11 -> M1 contributes to C00, C11
    
    # M2
    U_222[:, 1] = [0, 1] # A10+A11? No, Strassen M2 is (A10+A11)B00
    # My previous derivation:
    # M2: U=[0, 1] (A10+A11 is [0,1,1,1] in flat 2x2? No. A is 2x2.
    # Let's use the explicit flat indices from before.
    # A flat: [A00, A01, A10, A11]
    # U_222 is 2x7? No, A is 2x2, so 4 elements.
    # The local U for 2x2x2 should be 4x7.
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # M1
    U_222[:, 0] = [1, 0, 0, 1]
    V_222[:, 0] = [1, 0, 0, 1]
    W_222[:, 0] = [1, 0, 0, 1]
    
    # M2
    U_222[:, 1] = [0, 0, 1, 1]
    V_222[:, 1] = [1, 0, 0, 0]
    W_222[:, 1] = [0, 0, 1, -1]
    
    # M3
    U_222[:, 2] = [1, 0, 0, 0]
    V_222[:, 2] = [0, 0, 1, -1]
    W_222[:, 2] = [0, 1, 0, 1]
    
    # M4
    U_222[:, 3] = [0, 0, 0, 1]
    V_222[:, 3] = [1, -1, 0, 0]
    W_222[:, 3] = [1, 0, 1, 0]
    
    # M5
    U_222[:, 4] = [1, 1, 0, 0]
    V_222[:, 4] = [0, 0, 0, 1]
    W_222[:, 4] = [-1, 1, 0, 0]
    
    # M6
    U_222[:, 5] = [-1, 0, 1, 0]
    V_222[:, 5] = [1, 1, 0, 0]
    W_222[:, 5] = [0, 0, 0, 1]
    
    # M7
    U_222[:, 6] = [0, 1, 0, -1]
    V_222[:, 6] = [0, 0, 1, 1]
    W_222[:, 6] = [1, 0, 0, 0]

    # 2. Build 3x2x2 (Rank 11)
    # A: 3x2 (6), B: 2x2 (4), C: 3x2 (6)
    U_322 = np.zeros((6, 11))
    V_322 = np.zeros((4, 11))
    W_322 = np.zeros((6, 11))
    
    # Terms 0-6: Strassen for top 2x2 of A (rows 0,1) and C (rows 0,1)
    # Local A indices for top 2x2: 0,1,2,3
    # Local C indices for top 2x2: 0,1,2,3
    U_322[0:4, 0:7] = U_222
    V_322[:, 0:7] = V_222
    W_322[0:4, 0:7] = W_222
    
    # Terms 7-10: Naive for bottom 1x2 of A (row 2) and C (row 2)
    # A row 2 indices: 4, 5
    # C row 2 indices: 4, 5
    # C4 = A4*B0 + A5*B2
    # C5 = A4*B1 + A5*B3
    
    U_322[4, 7] = 1
    U_322[5, 8] = 1
    U_322[4, 9] = 1
    U_322[5, 10] = 1
    
    V_322[0, 7] = 1
    V_322[2, 8] = 1
    V_322[1, 9] = 1
    V_322[3, 10] = 1
    
    W_322[4, 7] = 1
    W_322[4, 8] = 1
    W_322[5, 9] = 1
    W_322[5, 10] = 1

    # 3. Build 3x2x6 (Rank 33)
    # A: 3x2 (6), B: 2x6 (12), C: 3x6 (18)
    U_326 = np.zeros((6, 33))
    V_326 = np.zeros((12, 33))
    W_326 = np.zeros((18, 33))
    
    for block in range(3):
        offset = block * 11
        # A is same
        U_326[:, offset:offset+11] = U_322
        
        # B block: cols block*2 to block*2+1
        # B is 2x6. Flat index j*6+k.
        # Block 0: k=0,1. j=0,1.
        # Local B indices 0,1,2,3 map to Global B indices:
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # j=1, k=0 -> 6
        # j=1, k=1 -> 7
        
        v_map = [block*2, block*2+1, 6+block*2, 6+block*2+1]
        for local_idx, global_idx in enumerate(v_map):
            V_326[global_idx, offset:offset+11] = V_322[local_idx, :]
            
        # C block: cols block*2 to block*2+1
        # C is 3x6. Flat index k*3+i.
        # Block 0: k=0,1.
        # Local C indices 0,1,2,3,4,5 map to Global C indices:
        # i=0, k=0 -> 0
        # i=1, k=0 -> 3
        # i=2, k=0 -> 6
        # i=0, k=1 -> 1
        # i=1, k=1 -> 4
        # i=2, k=1 -> 7
        
        w_map = [
            0*3 + 0, # i=0, k=0
            0*3 + 1, # i=1, k=0 -- Wait, C is k*N+i.
            # Local C in 3x2x2 is 3x2.
            # Local C index c_local = k_local * 3 + i.
            # Global C index c_global = k_global * 3 + i.
            # k_global = block + k_local.
            # So c_global = (block + k_local) * 3 + i = block*3 + k_local*3 + i = block*3 + c_local.
            # So just add block*3 to all local C indices.
        ]
        
        # Correction:
        # Local C indices are 0..5.
        # Global C indices are block*3 + local_c.
        for local_idx in range(6):
            global_idx = block * 3 + local_idx
            W_326[global_idx, offset:offset+11] = W_322[local_idx, :]

    # 4. Build 3x4x6 (Rank 66)
    # A: 3x4 (12), B: 4x6 (24), C: 3x6 (18)
    
    for block in range(2):
        offset = block * 33
        
        # A block: cols block*2 to block*2+1
        # A is 3x4. Flat index i*4+j.
        # Local A indices 0..5 map to Global A indices:
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 4
        # i=1, j=1 -> 5
        # i=2, j=0 -> 8
        # i=2, j=1 -> 9
        
        u_map = [
            0*4 + block*2,
            0*4 + block*2 + 1,
            1*4 + block*2,
            1*4 + block*2 + 1,
            2*4 + block*2,
            2*4 + block*2 + 1
        ]
        for local_idx, global_idx in enumerate(u_map):
            U[global_idx, offset:offset+33] = U_326[local_idx, :]
            
        # B block: rows block*2 to block*2+1
        # B is 4x6. Flat index j*6+k.
        # Local B indices 0..11 map to Global B indices:
        # j=0, k=0 -> 0
        # ...
        # j=1, k=5 -> 11
        # Global j = block*2 + local_j.
        # Global B index = (block*2 + local_j) * 6 + local_k.
        # Local B index = local_j * 6 + local_k.
        
        for local_j in range(2):
            for local_k in range(6):
                local_idx = local_j * 6 + local_k
                global_idx = (block * 2 + local_j) * 6 + local_k
                V[global_idx, offset:offset+33] = V_326[local_idx, :]
                
        # C is same for both blocks (summed)
        W[:, offset:offset+33] = W_326

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
