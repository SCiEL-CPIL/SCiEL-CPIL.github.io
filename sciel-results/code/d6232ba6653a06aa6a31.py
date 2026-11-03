"""Program for finding rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 2x3x8 matrix multiplication tensor using a block-Strassen approach.
    
    Strategy:
    - Split the 3x8 matrix B into two 3x4 blocks: B1 (cols 0-3), B2 (cols 4-7)
    - C1 = A * B1 (2x4), C2 = A * B2 (2x4)
    - Decompose each 2x3x4 multiplication using rank 22 construction.
    
    For 2x3x4:
    - Split A (2x3) into A_left (2x2, cols 0-1) and A_right (2x1, col 2)
    - Split B (3x4) into B_top (2x4, rows 0-1) and B_bottom (1x4, row 2)
    - C = A_left * B_top + A_right * B_bottom
    
    For A_left (2x2) * B_top (2x4):
    - Split B_top into two 2x2 blocks: B_tl (2x2, cols 0-1), B_tr (2x2, cols 2-3)
    - A_left * B_tl: 2x2x2 (rank 7 with Strassen)
    - A_left * B_tr: 2x2x2 (rank 7 with Strassen)
    Total: 14
    
    For A_right (2x1) * B_bottom (1x4):
    - This is 2x1x4, which requires 2*1*4 = 8 multiplications (schoolbook)
    
    Total rank per 2x3x4 block: 14 + 8 = 22
    Total rank for 2x3x8: 2 * 22 = 44
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b01, b10, b11]
        # c_idx: [c00, c01, c10, c11]
        
        # P1 = (a00 + a11) * (b00 + b11)
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # P2 = (a10 + a11) * b00
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        W[c_idx[3]][col] -= 1
        col += 1
        
        # P3 = a00 * (b10 - b11)
        U[a_idx[0]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] -= 1
        W[c_idx[1]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # P4 = a11 * (b01 - b00)
        U[a_idx[3]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[0]][col] -= 1
        W[c_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # P5 = (a00 + a01) * b11
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] -= 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # P6 = (a10 - a00) * (b00 + b01)
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # P7 = (a01 - a11) * (b10 + b11)
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[1]][col] -= 1
        col += 1

    def add_outer_block(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: [a0, a1] (2x1)
        # b_idx: [b0, b1, b2, b3] (1x4)
        # c_idx: [c00, c01, c02, c03, c10, c11, c12, c13] (2x4)
        
        # a0 * b0 -> c00
        U[a_idx[0]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1
        
        # a0 * b1 -> c01
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # a0 * b2 -> c02
        U[a_idx[0]][col] += 1
        V[b_idx[2]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # a0 * b3 -> c03
        U[a_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # a1 * b0 -> c10
        U[a_idx[1]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[4]][col] += 1
        col += 1
        
        # a1 * b1 -> c11
        U[a_idx[1]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[5]][col] += 1
        col += 1
        
        # a1 * b2 -> c12
        U[a_idx[1]][col] += 1
        V[b_idx[2]][col] += 1
        W[c_idx[6]][col] += 1
        col += 1
        
        # a1 * b3 -> c13
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[7]][col] += 1
        col += 1

    # Process Block 1 (Cols 0-3 of B, Cols 0-3 of C)
    # A indices are fixed:
    # A_left (2x2): A[0,0], A[0,1], A[1,0], A[1,1] -> U indices: 0, 1, 3, 4
    a_left = [0, 1, 3, 4]
    # A_right (2x1): A[0,2], A[1,2] -> U indices: 2, 5
    a_right = [2, 5]
    
    # B1 indices (3x4):
    # B_top (2x4): Rows 0,1. Cols 0-3.
    # B_tl (2x2): B[0,0], B[0,1], B[1,0], B[1,1] -> V indices: 0, 1, 8, 9
    b1_tl = [0, 1, 8, 9]
    # B_tr (2x2): B[0,2], B[0,3], B[1,2], B[1,3] -> V indices: 2, 3, 10, 11
    b1_tr = [2, 3, 10, 11]
    # B_bottom (1x4): Row 2. Cols 0-3.
    # B[2,0], B[2,1], B[2,2], B[2,3] -> V indices: 16, 17, 18, 19
    b1_bot = [16, 17, 18, 19]
    
    # C1 indices (2x4):
    # C[0,0], C[0,1], C[0,2], C[0,3], C[1,0], C[1,1], C[1,2], C[1,3]
    # W indices: k*n + i.
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=1, i=0 -> 2
    # C[0,2] -> k=2, i=0 -> 4
    # C[0,3] -> k=3, i=0 -> 6
    # C[1,0] -> k=0, i=1 -> 1
    # C[1,1] -> k=1, i=1 -> 3
    # C[1,2] -> k=2, i=1 -> 5
    # C[1,3] -> k=3, i=1 -> 7
    c1_tl = [0, 2, 1, 3] # c00, c01, c10, c11
    c1_tr = [4, 6, 5, 7] # c02, c03, c12, c13
    c1_all = [0, 2, 4, 6, 1, 3, 5, 7]
    
    add_strassen_block(a_left, b1_tl, c1_tl)
    add_strassen_block(a_left, b1_tr, c1_tr)
    add_outer_block(a_right, b1_bot, c1_all)
    
    # Process Block 2 (Cols 4-7 of B, Cols 4-7 of C)
    # B2 indices:
    # B_tl (2x2): B[0,4], B[0,5], B[1,4], B[1,5] -> V indices: 4, 5, 12, 13
    b2_tl = [4, 5, 12, 13]
    # B_tr (2x2): B[0,6], B[0,7], B[1,6], B[1,7] -> V indices: 6, 7, 14, 15
    b2_tr = [6, 7, 14, 15]
    # B_bottom (1x4): B[2,4], B[2,5], B[2,6], B[2,7] -> V indices: 20, 21, 22, 23
    b2_bot = [20, 21, 22, 23]
    
    # C2 indices:
    # C[0,4] -> k=4, i=0 -> 8
    # C[0,5] -> k=5, i=0 -> 10
    # C[0,6] -> k=6, i=0 -> 12
    # C[0,7] -> k=7, i=0 -> 14
    # C[1,4] -> k=4, i=1 -> 9
    # C[1,5] -> k=5, i=1 -> 11
    # C[1,6] -> k=6, i=1 -> 13
    # C[1,7] -> k=7, i=1 -> 15
    c2_tl = [8, 10, 9, 11]
    c2_tr = [12, 14, 13, 15]
    c2_all = [8, 10, 12, 14, 9, 11, 13, 15]
    
    add_strassen_block(a_left, b2_tl, c2_tl)
    add_strassen_block(a_left, b2_tr, c2_tr)
    add_outer_block(a_right, b2_bot, c2_all)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
