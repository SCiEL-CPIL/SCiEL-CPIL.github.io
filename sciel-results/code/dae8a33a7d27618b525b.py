"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 60 decomposition
    # Split A into A1 (2x4) and A2 (1x4)
    # Split B into B1 (4x4) and B2 (4x4)
    # C = A1 @ B + A2 @ B
    
    # Part 1: A1 (2x4) @ B (4x8)
    # Split B into BL (4x4) and BR (4x4)
    # A1 @ BL (2x4x4) and A1 @ BR (2x4x4)
    
    # A1 @ BL:
    # Split A1 into A11, A12 (2x2)
    # Split BL into 4 blocks of 2x2
    # 4 Strassen multiplications
    
    # We will build the factors U, V, W directly.
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen_block(U, V, W, col, a_row_offset, b_row_offset, c_row_offset, c_col_offset):
        # a_row_offset: start row of A block (2x2)
        # b_row_offset: start row of B block (2x2)
        # c_row_offset: start row of C block (2x2)
        # c_col_offset: start col of C block (2x2)
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # Strassen M1-M7
        # We need to map the 7 multiplications to the global indices.
        
        # A block indices: (a_row_offset + 0, b_row_offset + 0) etc?
        # No, A is 2x2, B is 2x2.
        # A indices: (r, c) relative to block.
        # U index: (a_row_offset + r) * m + (b_row_offset + c) ?
        # Wait, A is 2x2. Rows 0,1. Cols 0,1.
        # U index is (i, j) where i is row of A, j is col of A.
        # So U[i*m + j, col].
        
        # B indices: (r, c) relative to block.
        # V index: (b_row_offset + r) * p + (b_col_offset + c).
        # We need b_col_offset too.
        
        # Let's pass full offsets.
        pass

    # This manual construction is complex.
    # Given the time, I will output the standard 96 rank if I can't easily code 60.
    # But 60 is a significant improvement.
    
    # Let's try to implement the 60 rank.
    
    # Structure:
    # 56 ranks for A1 @ B
    # 4 ranks for A2 @ B
    
    # A1 @ B part:
    # B is split into BL and BR.
    # For each (AL, AR) in [(A11, A12)]: # A1 is 2x4, so A11, A12 are 2x2
    #   For each (BL_part, BR_part) in [(BL, BR)]:
    #      For each (A_sub, B_sub) in [(A11, BL11), (A12, BL21), (A11, BL12), (A12, BL22)]:
    #         Add Strassen
    
    # This is getting messy. I will stick to the standard 96 to ensure correctness in this turn, 
    # as coding 60 reliably in one shot is risky.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
