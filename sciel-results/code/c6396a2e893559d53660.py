import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Rank 49 Strassen decomposition for 4x4x4
    # We construct U, V, W for 4x4x4, then duplicate for 4x4x8
    
    # Step 1: Define Strassen 2x2x2 (Rank 7)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # M1
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # M2
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # M3
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # M4
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = -1; W_2[2, 3] = 1
    
    # M5
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = 1; W_2[1, 4] = 1
    
    # M6
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # M7
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = -1

    # Step 2: Build 4x4x4 using 7 block multiplications of 2x2x2
    # U_4 shape (16, 49), V_4 shape (16, 49), W_4 shape (16, 49)
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    # Strassen coefficients for 4x4x4 (treating 4x4 as 2x2 blocks)
    # A blocks: A00(0-3), A01(4-7), A10(8-11), A11(12-15)
    # B blocks: B00(0-3), B01(4-7), B10(8-11), B11(12-15)
    # C blocks: C00(0-3), C01(4-7), C10(8-11), C11(12-15)
    
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    
    # C00 = P0 - P3 + P4 - P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    block_specs = [
        # (U_blocks, V_blocks, W_blocks)
        # P0
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, -1)]),
        # P1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # P2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # P3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, -1), (2, 1)]),
        # P4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),
        # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, -1)]),
    ]
    
    # Mapping from block index to global row offset
    # Block 0: 0-3, Block 1: 4-7, Block 2: 8-11, Block 3: 12-15
    def get_offset(block_idx):
        return block_idx * 4
    
    for k in range(7):
        u_blocks, v_blocks, w_blocks = block_specs[k]
        
        # Populate U_4
        for b_idx, coeff in u_blocks:
            offset = get_offset(b_idx)
            for r in range(4):
                for c in range(7):
                    if U_2[r, c] != 0:
                        U_4[offset + r, k * 7 + c] += coeff * U_2[r, c]
                        
        # Populate V_4
        for b_idx, coeff in v_blocks:
            offset = get_offset(b_idx)
            for r in range(4):
                for c in range(7):
                    if V_2[r, c] != 0:
                        V_4[offset + r, k * 7 + c] += coeff * V_2[r, c]
                        
        # Populate W_4
        for b_idx, coeff in w_blocks:
            offset = get_offset(b_idx)
            for r in range(4):
                for c in range(7):
                    if W_2[r, c] != 0:
                        W_4[offset + r, k * 7 + c] += coeff * W_2[r, c]

    # Step 3: Combine two 4x4x4 into 4x4x8
    # U is shared (16x49) -> (16x98)
    # V is split (16x49) for B_L, (16x49) for B_R -> (16x98)
    # W is split (16x49) for C_L, (16x49) for C_R -> (16x98)
    
    U = np.zeros((16, 98))
    V = np.zeros((16, 98))
    W = np.zeros((16, 98))
    
    # First half (cols 0-48)
    U[:, 0:49] = U_4
    V[:, 0:49] = V_4
    W[:, 0:49] = W_4
    
    # Second half (cols 49-97)
    U[:, 49:98] = U_4
    V[:, 49:98] = V_4
    W[:, 49:98] = W_4
    
    # Convert to lists for return
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
