import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Base case: 2x2x2 Strassen
    # U, V, W are 4x7
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # M1
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # M2
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1; W_222[3, 1] = 1
    
    # M3
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1; W_222[3, 2] = 1
    
    # M4
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1; W_222[2, 3] = 1
    
    # M5
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = -1; W_222[1, 4] = 1
    
    # M6
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[3, 5] = 1
    
    # M7
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1; W_222[3, 6] = -1

    # Helper to place a block
    def place(U, V, W, u_block, v_block, w_block, u_off, v_off, w_off, col_off):
        U[u_off:u_off+u_block.shape[0], col_off:col_off+u_block.shape[1]] += u_block
        V[v_off:v_off+v_block.shape[0], col_off:col_off+v_block.shape[1]] += v_block
        W[w_off:w_off+w_block.shape[0], col_off:col_off+w_block.shape[1]] += w_block

    # Build 2x2x4 (Rank 14)
    # A is 2x2 (4 elems), B is 2x4 (8 elems), C is 2x4 (8 elems)
    # Split B into B_L (cols 0-1) and B_R (cols 2-3)
    # Split C into C_L (cols 0-1) and C_R (cols 2-3)
    U_224 = np.zeros((4, 14))
    V_224 = np.zeros((8, 14))
    W_224 = np.zeros((8, 14))
    
    # Part 1: A * B_L -> C_L
    place(U_224, V_224, W_224, U_222, V_222, W_222, 0, 0, 0, 0)
    # Part 2: A * B_R -> C_R
    place(U_224, V_224, W_224, U_222, V_222, W_222, 0, 4, 4, 7)

    # Build 2x2x8 (Rank 28)
    # A is 2x2, B is 2x8 (16 elems), C is 2x8 (16 elems)
    # Split B into B_L (cols 0-3) and B_R (cols 4-7)
    # Split C into C_L (cols 0-3) and C_R (cols 4-7)
    U_228 = np.zeros((4, 28))
    V_228 = np.zeros((16, 28))
    W_228 = np.zeros((16, 28))
    
    # Part 1: A * B_L -> C_L
    place(U_228, V_228, W_228, U_224, V_224, W_224, 0, 0, 0, 0)
    # Part 2: A * B_R -> C_R
    place(U_228, V_228, W_228, U_224, V_224, W_224, 0, 8, 8, 14)

    # Build 2x4x8 (Rank 56)
    # A is 2x4 (8 elems), B is 4x8 (32 elems), C is 2x8 (16 elems)
    # Split A into A_L (cols 0-1) and A_R (cols 2-3)
    # Split B into B_T (rows 0-1) and B_B (rows 2-3)
    # C = A_L * B_T + A_R * B_B
    
    U_final = np.zeros((8, 56))
    V_final = np.zeros((32, 56))
    W_final = np.zeros((16, 56))
    
    # Part 1: A_L * B_T -> C_1
    # A_L is rows 0-3 of A_final (indices 0-3)
    # B_T is rows 0-15 of B_final (indices 0-15)
    # C_1 contributes to all of C_final
    place(U_final, V_final, W_final, U_228, V_228, W_228, 0, 0, 0, 0)
    
    # Part 2: A_R * B_B -> C_2
    # A_R is rows 4-7 of A_final (indices 4-7)
    # B_B is rows 16-31 of B_final (indices 16-31)
    # C_2 contributes to all of C_final
    place(U_final, V_final, W_final, U_228, V_228, W_228, 4, 16, 0, 28)

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

