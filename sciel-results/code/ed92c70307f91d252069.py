import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Rank: 56.
    """
    n, m, p = DIMENSIONS
    
    # Define the rank
    R = 56
    
    # Initialize U, V, W with zeros
    U = np.zeros((R, n * m))
    V = np.zeros((R, m * p))
    W = np.zeros((R, p * n))
    
    rank_idx = 0
    
    # Helper to add a block
    def add_block(u_indices, v_indices, w_indices, U_block, V_block, W_block):
        nonlocal rank_idx
        for r in range(U_block.shape[1]):
            for i, u_idx in enumerate(u_indices):
                U[rank_idx, u_idx] = U_block[i, r]
            for i, v_idx in enumerate(v_indices):
                V[rank_idx, v_idx] = V_block[i, r]
            for i, w_idx in enumerate(w_indices):
                W[rank_idx, w_idx] = W_block[i, r]
            rank_idx += 1

    # Strassen 2x2x2
    def strassen_2x2x2():
        U_b = np.zeros((4, 7))
        V_b = np.zeros((4, 7))
        W_b = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U_b[0, 0] = 1; U_b[3, 0] = 1
        V_b[0, 0] = 1; V_b[3, 0] = 1
        W_b[0, 0] = 1; W_b[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U_b[2, 1] = 1; U_b[3, 1] = 1
        V_b[0, 1] = 1
        W_b[2, 1] = 1; W_b[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U_b[0, 2] = 1
        V_b[1, 2] = 1; V_b[3, 2] = -1
        W_b[1, 2] = 1; W_b[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U_b[3, 3] = 1
        V_b[2, 3] = 1; V_b[0, 3] = -1
        W_b[0, 3] = 1; W_b[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U_b[0, 4] = 1; U_b[1, 4] = 1
        V_b[3, 4] = 1
        W_b[0, 4] = -1; W_b[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U_b[2, 5] = 1; U_b[0, 5] = -1
        V_b[0, 5] = 1; V_b[1, 5] = 1
        W_b[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U_b[1, 6] = 1; U_b[3, 6] = -1
        V_b[2, 6] = 1; V_b[3, 6] = 1
        W_b[0, 6] = 1
        
        return U_b, V_b, W_b

    # Schoolbook 2x2x1
    def schoolbook_2x2x1():
        U_b = np.zeros((4, 4))
        V_b = np.zeros((2, 4))
        W_b = np.zeros((2, 4))
        U_b[0, 0] = 1; V_b[0, 0] = 1; W_b[0, 0] = 1
        U_b[1, 1] = 1; V_b[1, 1] = 1; W_b[0, 1] = 1
        U_b[2, 2] = 1; V_b[0, 2] = 1; W_b[1, 2] = 1
        U_b[3, 3] = 1; V_b[1, 3] = 1; W_b[1, 3] = 1
        return U_b, V_b, W_b

    # Schoolbook 1x2x2
    def schoolbook_1x2x2():
        U_b = np.zeros((2, 4))
        V_b = np.zeros((4, 4))
        W_b = np.zeros((2, 4))
        U_b[0, 0] = 1; V_b[0, 0] = 1; W_b[0, 0] = 1
        U_b[1, 1] = 1; V_b[2, 1] = 1; W_b[0, 1] = 1
        U_b[0, 2] = 1; V_b[1, 2] = 1; W_b[1, 2] = 1
        U_b[1, 3] = 1; V_b[3, 3] = 1; W_b[1, 3] = 1
        return U_b, V_b, W_b

    # Schoolbook 1x2x1
    def schoolbook_1x2x1():
        U_b = np.zeros((2, 2))
        V_b = np.zeros((2, 2))
        W_b = np.zeros((1, 2))
        U_b[0, 0] = 1; V_b[0, 0] = 1; W_b[0, 0] = 1
        U_b[1, 1] = 1; V_b[1, 1] = 1; W_b[0, 1] = 1
        return U_b, V_b, W_b

    # Indices mapping
    # U: i*m + j
    # V: i*p + j
    # W: k*n + i
    
    # Part 1: A rows 0,1 (2x4) * B (4x5)
    # Split B cols into 2+2+1? No, split B rows into 2+2.
    # A1_top (2x2) * B_top (2x5)
    # A1_bot (2x2) * B_bot (2x5)
    
    # A1_top: U indices 0,1, 4,5
    # A1_bot: U indices 2,3, 6,7
    
    # B_top: V indices 0,1, 5,6 (cols 0,1) -> 2x2 block
    #         V indices 2,3, 7,8 (cols 2,3) -> 2x2 block
    #         V indices 4,9 (col 4) -> 2x1 block
    
    # B_bot: V indices 10,11, 15,16 (cols 0,1)
    #         V indices 12,13, 17,18 (cols 2,3)
    #         V indices 14,19 (col 4)
    
    # W indices for C rows 0,1
    # Row 0: 0, 3, 6, 9, 12
    # Row 1: 1, 4, 7, 10, 13
    
    # Block 1: A1_top * B_top_left (2x2x2)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], U_b, V_b, W_b)
    
    # Block 2: A1_top * B_top_mid (2x2x2)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], U_b, V_b, W_b)
    
    # Block 3: A1_top * B_top_right (2x2x1)
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block([0,1,4,5], [4,9], [12,13], U_b, V_b, W_b)
    
    # Block 4: A1_bot * B_bot_left (2x2x2)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], U_b, V_b, W_b)
    
    # Block 5: A1_bot * B_bot_mid (2x2x2)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], U_b, V_b, W_b)
    
    # Block 6: A1_bot * B_bot_right (2x2x1)
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block([2,3,6,7], [14,19], [12,13], U_b, V_b, W_b)
    
    # Part 2: A row 2 (1x4) * B (4x5)
    # A2_top: U indices 8,9
    # A2_bot: U indices 10,11
    
    # W indices for C row 2
    # Row 2: 2, 5, 8, 11, 14
    
    # Block 7: A2_top * B_top_left (1x2x2)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block([8,9], [0,1,5,6], [2,5], U_b, V_b, W_b)
    
    # Block 8: A2_top * B_top_mid (1x2x2)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block([8,9], [2,3,7,8], [8,11], U_b, V_b, W_b)
    
    # Block 9: A2_top * B_top_right (1x2x1)
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_block([8,9], [4,9], [14], U_b, V_b, W_b)
    
    # Block 10: A2_bot * B_bot_left (1x2x2)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block([10,11], [10,11,15,16], [2,5], U_b, V_b, W_b)
    
    # Block 11: A2_bot * B_bot_mid (1x2x2)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block([10,11], [12,13,17,18], [8,11], U_b, V_b, W_b)
    
    # Block 12: A2_bot * B_bot_right (1x2x1)
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_block([10,11], [14,19], [14], U_b, V_b, W_b)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
