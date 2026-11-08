import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Approach: Split N (rows of A / rows of C) into 1 and 2.
    
    C (3x5) = [C_top (1x5); C_bot (2x5)]
    A (3x4) = [A_top (1x4); A_bot (2x4)]
    
    C_top = A_top * B  (1x4 * 4x5) -> Rank 20 (5 dot products of length 4)
    C_bot = A_bot * B  (2x4 * 4x5) -> Rank 36 (2 * Rank(2x4x5))
    
    Rank(2x4x5):
    Split M (cols of A / rows of B) into 2 and 2.
    A_bot = [A_bot1 (2x2), A_bot2 (2x2)]
    B = [B1 (2x5); B2 (2x5)]
    C_bot = A_bot1 * B1 + A_bot2 * B2
    Each term is 2x2x5.
    
    Rank(2x2x5):
    Split P (cols of B / cols of C) into 2, 2, 1.
    B_j = [B_j1 (2x2), B_j2 (2x2), B_j3 (2x1)]
    A_i * B_j = A_i * B_j1 + A_i * B_j2 + A_i * B_j3
    Rank(2x2x2) = 7 (Strassen)
    Rank(2x2x1) = 4 (Schoolbook)
    Rank(2x2x5) = 7 + 7 + 4 = 18.
    
    Total Rank = 20 + 2 * 18 = 56.
    """
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_indices, v_indices, w_indices, U_blk, V_blk, W_blk):
        rank = U_blk.shape[1]
        for r in range(rank):
            u_vec = np.zeros(n * m)
            for i, idx in enumerate(u_indices):
                u_vec[idx] = U_blk[i, r]
            U_parts.append(u_vec)
            
            v_vec = np.zeros(m * p)
            for i, idx in enumerate(v_indices):
                v_vec[idx] = V_blk[i, r]
            V_parts.append(v_vec)
            
            w_vec = np.zeros(p * n)
            for i, idx in enumerate(w_indices):
                w_vec[idx] = W_blk[i, r]
            W_parts.append(w_vec)

    # --- Helper Decompositions ---
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1; V[2, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
        return U, V, W

    def schoolbook_1x4x1():
        # Rank 4
        U = np.zeros((4, 4))
        V = np.zeros((4, 4))
        W = np.zeros((1, 4))
        for k in range(4):
            U[k, k] = 1
            V[k, k] = 1
            W[0, k] = 1
        return U, V, W

    def schoolbook_2x2x1():
        # Rank 4
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        # C0 = A00*B0 + A01*B1
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # C1 = A10*B0 + A11*B1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W

    # --- Assembly ---
    
    # 1. C_top (Row 0 of C) = A_top (Row 0 of A) * B
    # A_top is 1x4. B is 4x5.
    # C_top has 5 elements.
    # For each column k of B (0..4), we do a dot product of length 4.
    # U indices for A_top: [0, 1, 2, 3] (Row 0 of A)
    # V indices for B column k: [k, 5+k, 10+k, 15+k] (Column k of B)
    # W indices for C_top column k: k*n + 0 = k*3 + 0 = k*3
    
    U_top_idx = [0, 1, 2, 3]
    for k in range(5):
        V_col_idx = [k, 5+k, 10+k, 15+k]
        W_idx = [k*3] # C[0, k] is at index k*3 + 0
        
        U_blk, V_blk, W_blk = schoolbook_1x4x1()
        add_block(U_top_idx, V_col_idx, W_idx, U_blk, V_blk, W_blk)
        
    # 2. C_bot (Rows 1,2 of C) = A_bot (Rows 1,2 of A) * B
    # A_bot is 2x4. B is 4x5.
    # Split M into 2+2.
    # A_bot1 (Rows 1,2; Cols 0,1)
    # A_bot2 (Rows 1,2; Cols 2,3)
    
    # U indices for A_bot1: [4, 5, 8, 9] (A[1,0], A[1,1], A[2,0], A[2,1])
    # U indices for A_bot2: [6, 7, 10, 11] (A[1,2], A[1,3], A[2,2], A[2,3])
    
    # B1 (Rows 0,1)
    # B2 (Rows 2,3)
    
    # Term 2.1: A_bot1 * B1 (2x2x5)
    # Split P into 2, 2, 1.
    # B1_1 (Cols 0,1): V [0, 1, 5, 6]
    # B1_2 (Cols 2,3): V [2, 3, 7, 8]
    # B1_3 (Col 4):    V [4, 9]
    
    # C_bot cols 0,1: W indices for C[1,0], C[1,1], C[2,0], C[2,1]
    # C[1,0] -> 0*3+1 = 1
    # C[1,1] -> 1*3+1 = 4
    # C[2,0] -> 0*3+2 = 2
    # C[2,1] -> 1*3+2 = 5
    # W_idx_01 = [1, 4, 2, 5]
    
    # C_bot cols 2,3: W indices for C[1,2], C[1,3], C[2,2], C[2,3]
    # C[1,2] -> 2*3+1 = 7
    # C[1,3] -> 3*3+1 = 10
    # C[2,2] -> 2*3+2 = 8
    # C[2,3] -> 3*3+2 = 11
    # W_idx_23 = [7, 10, 8, 11]
    
    # C_bot col 4: W indices for C[1,4], C[2,4]
    # C[1,4] -> 4*3+1 = 13
    # C[2,4] -> 4*3+2 = 14
    # W_idx_4 = [13, 14]
    
    U_bot1_idx = [4, 5, 8, 9]
    
    # 2.1.1: A_bot1 * B1_1 (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_bot1_idx, [0, 1, 5, 6], [1, 4, 2, 5], U_blk, V_blk, W_blk)
    
    # 2.1.2: A_bot1 * B1_2 (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_bot1_idx, [2, 3, 7, 8], [7, 10, 8, 11], U_blk, V_blk, W_blk)
    
    # 2.1.3: A_bot1 * B1_3 (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block(U_bot1_idx, [4, 9], [13, 14], U_blk, V_blk, W_blk)
    
    # Term 2.2: A_bot2 * B2 (2x2x5)
    # B2_1 (Cols 0,1): V [10, 11, 15, 16]
    # B2_2 (Cols 2,3): V [12, 13, 17, 18]
    # B2_3 (Col 4):    V [14, 19]
    
    U_bot2_idx = [6, 7, 10, 11]
    
    # 2.2.1: A_bot2 * B2_1 (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_bot2_idx, [10, 11, 15, 16], [1, 4, 2, 5], U_blk, V_blk, W_blk)
    
    # 2.2.2: A_bot2 * B2_2 (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_bot2_idx, [12, 13, 17, 18], [7, 10, 8, 11], U_blk, V_blk, W_blk)
    
    # 2.2.3: A_bot2 * B2_3 (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block(U_bot2_idx, [14, 19], [13, 14], U_blk, V_blk, W_blk)

    # Total Rank:
    # C_top: 5 * 4 = 20
    # C_bot: 2 * (7 + 7 + 4) = 36
    # Total: 56
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
