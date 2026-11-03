import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Two j-blocks (size 2 each).
    # For each j-block:
    #   One i-row (index 2): rank 2.
    #   Two i-rows (indices 0,1): rank 18 (7+7+4).
    # Total per j-block: 20.
    # Total rank: 40.
    rank = 40
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen helper
    def add_strassen(i_offset, j_offset, k_offset, start_col):
        # Adds 7 terms for 2x2 matrix mult
        # A is 2x2 at (i_offset, j_offset)
        # B is 2x2 at (j_offset, k_offset)
        # C is 2x2 at (i_offset, k_offset)
        nonlocal col
        # p1 = (x00+x11)(y00+y11)
        U[(i_offset + 0) * m + (j_offset + 0)][start_col] = 1
        U[(i_offset + 1) * m + (j_offset + 1)][start_col] = 1
        V[(j_offset + 0) * p + (k_offset + 0)][start_col] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][start_col] = 1
        W[(k_offset + 0) * n + (i_offset + 0)][start_col] = 1
        W[(k_offset + 1) * n + (i_offset + 1)][start_col] = 1
        W[(k_offset + 0) * n + (i_offset + 1)][start_col] = 1
        W[(k_offset + 1) * n + (i_offset + 0)][start_col] = 1
        col += 1
        
        # p2 = (x10+x11)y00
        U[(i_offset + 1) * m + (j_offset + 0)][start_col] = 1
        U[(i_offset + 1) * m + (j_offset + 1)][start_col] = 1
        V[(j_offset + 0) * p + (k_offset + 0)][start_col] = 1
        W[(k_offset + 0) * n + (i_offset + 1)][start_col] = 1
        W[(k_offset + 1) * n + (i_offset + 1)][start_col] = 1
        col += 1
        
        # p3 = x00(y01-y11)
        U[(i_offset + 0) * m + (j_offset + 0)][start_col] = 1
        V[(j_offset + 0) * p + (k_offset + 1)][start_col] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][start_col] = -1
        W[(k_offset + 1) * n + (i_offset + 0)][start_col] = 1
        W[(k_offset + 1) * n + (i_offset + 1)][start_col] = 1
        col += 1
        
        # p4 = x11(y10-y00)
        U[(i_offset + 1) * m + (j_offset + 1)][start_col] = 1
        V[(j_offset + 1) * p + (k_offset + 0)][start_col] = 1
        V[(j_offset + 0) * p + (k_offset + 0)][start_col] = -1
        W[(k_offset + 0) * n + (i_offset + 0)][start_col] = 1
        W[(k_offset + 0) * n + (i_offset + 1)][start_col] = 1
        col += 1
        
        # p5 = (x00+x10)y11
        U[(i_offset + 0) * m + (j_offset + 0)][start_col] = 1
        U[(i_offset + 1) * m + (j_offset + 0)][start_col] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][start_col] = 1
        W[(k_offset + 1) * n + (i_offset + 0)][start_col] = 1
        W[(k_offset + 1) * n + (i_offset + 1)][start_col] = 1
        col += 1
        
        # p6 = (x01-x00)(y00+y10)
        U[(i_offset + 0) * m + (j_offset + 1)][start_col] = 1
        U[(i_offset + 0) * m + (j_offset + 0)][start_col] = -1
        V[(j_offset + 0) * p + (k_offset + 0)][start_col] = 1
        V[(j_offset + 1) * p + (k_offset + 0)][start_col] = 1
        W[(k_offset + 0) * n + (i_offset + 1)][start_col] = 1
        W[(k_offset + 1) * n + (i_offset + 1)][start_col] = 1
        col += 1
        
        # p7 = (x11-x10)(y01+y11)
        U[(i_offset + 1) * m + (j_offset + 1)][start_col] = 1
        U[(i_offset + 1) * m + (j_offset + 0)][start_col] = -1
        V[(j_offset + 0) * p + (k_offset + 1)][start_col] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][start_col] = 1
        W[(k_offset + 0) * n + (i_offset + 0)][start_col] = 1
        W[(k_offset + 1) * n + (i_offset + 0)][start_col] = 1
        col += 1

    def add_schoolbook_2x2x1(i_offset, j_offset, k_offset, start_col):
        # Adds 4 terms for 2x2 by 2x1
        nonlocal col
        # a*b11 -> c0
        U[(i_offset + 0) * m + (j_offset + 0)][start_col] = 1
        V[(j_offset + 0) * p + (k_offset + 0)][start_col] = 1
        W[(k_offset + 0) * n + (i_offset + 0)][start_col] = 1
        col += 1
        # b*b21 -> c0
        U[(i_offset + 0) * m + (j_offset + 1)][start_col] = 1
        V[(j_offset + 1) * p + (k_offset + 0)][start_col] = 1
        W[(k_offset + 0) * n + (i_offset + 0)][start_col] = 1
        col += 1
        # c*b11 -> c1
        U[(i_offset + 1) * m + (j_offset + 0)][start_col] = 1
        V[(j_offset + 0) * p + (k_offset + 0)][start_col] = 1
        W[(k_offset + 0) * n + (i_offset + 1)][start_col] = 1
        col += 1
        # d*b21 -> c1
        U[(i_offset + 1) * m + (j_offset + 1)][start_col] = 1
        V[(j_offset + 1) * p + (k_offset + 0)][start_col] = 1
        W[(k_offset + 0) * n + (i_offset + 1)][start_col] = 1
        col += 1

    # Loop over j-blocks
    for j_block in range(2):
        j_off = j_block * 2
        
        # Handle i=2 (rank 2)
        # C_2k = A_20 B_0k + A_21 B_1k
        # Term 1
        U[2 * m + (j_off + 0)][col] = 1
        V[(j_off + 0) * p + 0][col] = 1 # k=0
        W[0 * n + 2][col] = 1
        V[(j_off + 0) * p + 1][col] = 1 # k=1
        W[1 * n + 2][col] = 1
        V[(j_off + 0) * p + 2][col] = 1 # k=2
        W[2 * n + 2][col] = 1
        V[(j_off + 0) * p + 3][col] = 1 # k=3
        W[3 * n + 2][col] = 1
        V[(j_off + 0) * p + 4][col] = 1 # k=4
        W[4 * n + 2][col] = 1
        col += 1
        
        # Term 2
        U[2 * m + (j_off + 1)][col] = 1
        V[(j_off + 1) * p + 0][col] = 1
        W[0 * n + 2][col] = 1
        V[(j_off + 1) * p + 1][col] = 1
        W[1 * n + 2][col] = 1
        V[(j_off + 1) * p + 2][col] = 1
        W[2 * n + 2][col] = 1
        V[(j_off + 1) * p + 3][col] = 1
        W[3 * n + 2][col] = 1
        V[(j_off + 1) * p + 4][col] = 1
        W[4 * n + 2][col] = 1
        col += 1
        
        # Handle i=0,1 (rank 18)
        i_off = 0
        
        # k=0,1 (Strassen)
        add_strassen(i_off, j_off, 0, col)
        
        # k=2,3 (Strassen)
        add_strassen(i_off, j_off, 2, col)
        
        # k=4 (Schoolbook 2x2x1)
        add_schoolbook_2x2x1(i_off, j_off, 4, col)

    return {"U": U, "V": V, "W": W}
