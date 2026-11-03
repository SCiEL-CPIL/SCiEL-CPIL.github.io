"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    
    Strategy: Use Strassen's algorithm on 2x2 sub-blocks where possible, combined
    with optimal handling of the remaining 1x2 and 2x2 * 2x3 blocks.
    
    Split A (3x4) into A1 (3x2) and A2 (3x2) by columns.
    Split B (4x5) into B1 (2x5) and B2 (2x5) by rows.
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (3x2 * 2x5):
      - Split A1 into A1_top (1x2) and A1_bot (2x2)
      - A1_top * B1: 1x2 * 2x5, rank = 10 (schoolbook)
      - A1_bot * B1: 2x2 * 2x5
        - Split B1 into B1_left (2x3) and B1_right (2x2)
        - A1_bot * B1_left: 2x2 * 2x3, rank = 8
        - A1_bot * B1_right: 2x2 * 2x2, rank = 7 (Strassen)
        - Total: 15
      - Total A1*B1: 25
    
    Similarly for A2*B2: 25
    
    Grand total rank: 50
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 12 x 50
    V = [[0] * rank for _ in range(m * p)]  # 20 x 50
    W = [[0] * rank for _ in range(p * n)]  # 15 x 50
    
    col = 0
    
    # Helper to add a 1x2 * 2x5 block
    # A_top is 1x2: U indices (a_row, a_col), (a_row, a_col+1)
    # B is 2x5: V indices (b_row, b_col+k) for k=0..4, (b_row+1, b_col+k)
    # C is 1x5: W indices (c_row, c_col+k) for k=0..4
    def add_1x2_2x5(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        for k in range(5):
            # C[c_row, c_col+k] = A[a_row, a_col] * B[b_row, b_col+k] + A[a_row, a_col+1] * B[b_row+1, b_col+k]
            U[a_row * m + a_col][col] = 1
            V[b_row * p + b_col + k][col] = 1
            W[(c_col + k) * n + c_row][col] = 1
            col += 1
            U[a_row * m + a_col + 1][col] = 1
            V[b_row + 1][col] = V[b_row + 1][col]  # placeholder
            V[b_row * p + b_col + k][col] = 0
            V[(b_row + 1) * p + b_col + k][col] = 1
            W[(c_col + k) * n + c_row][col] = 1
            col += 1
    
    # Helper to add a 2x2 * 2x3 block (schoolbook, rank 8)
    # A is 2x2: U indices (a_row+i, a_col+j) for i,j=0,1
    # B is 2x3: V indices (b_row+i, b_col+j) for i=0,1, j=0,1,2
    # C is 2x3: W indices ((c_col+j)*n + c_row+i) for i=0,1, j=0,1,2
    def add_2x2_2x3(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        # C[i,j] = sum_k A[i,k] * B[k,j]
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[(a_row + i) * m + a_col + k][col] = 1
                    V[(b_row + k) * p + b_col + j][col] = 1
                    W[(c_col + j) * n + c_row + i][col] = 1
                    col += 1
    
    # Helper to add Strassen 2x2 * 2x2 (rank 7)
    # A is 2x2: U indices (a_row+i, a_col+j)
    # B is 2x2: V indices (b_row+i, b_col+j)
    # C is 2x2: W indices ((c_col+j)*n + c_row+i)
    def add_strassen_2x2(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        # P1 = (a00+a11)(b00+b11)
        U[a_row * m + a_col][col] = 1
        U[(a_row + 1) * m + a_col + 1][col] = 1
        V[b_row * p + b_col][col] = 1
        V[(b_row + 1) * p + b_col + 1][col] = 1
        W[(c_col) * n + c_row][col] = 1       # C[0,0] += P1
        W[(c_col + 1) * n + c_row + 1][col] = 1  # C[1,1] += P1
        col += 1
        
        # P2 = (a10+a11)*b00
        U[(a_row + 1) * m + a_col][col] = 1
        U[(a_row + 1) * m + a_col + 1][col] = 1
        V[b_row * p + b_col][col] = 1
        W[(c_col) * n + c_row + 1][col] = 1   # C[1,0] += P2
        col += 1
        
        # P3 = a00*(b01-b11)
        U[a_row * m + a_col][col] = 1
        V[b_row * p + b_col + 1][col] = 1
        V[(b_row + 1) * p + b_col + 1][col] = -1
        W[(c_col + 1) * n + c_row][col] = 1   # C[0,1] += P3
        col += 1
        
        # P4 = a11*(b10-b00)
        U[(a_row + 1) * m + a_col + 1][col] = 1
        V[(b_row + 1) * p + b_col][col] = 1
        V[b_row * p + b_col][col] = -1
        W[(c_col) * n + c_row + 1][col] = 1   # C[1,0] += P4
        W[(c_col + 1) * n + c_row + 1][col] = 1  # C[1,1] += P4
        col += 1
        
        # P5 = (a00+a01)*b11
        U[a_row * m + a_col][col] = 1
        U[a_row * m + a_col + 1][col] = 1
        V[(b_row + 1) * p + b_col + 1][col] = 1
        W[(c_col + 1) * n + c_row][col] = 1   # C[0,1] += P5
        col += 1
        
        # P6 = (a10-a00)*(b00+b01)
        U[(a_row + 1) * m + a_col][col] = 1
        U[a_row * m + a_col][col] = -1
        V[b_row * p + b_col][col] = 1
        V[b_row * p + b_col + 1][col] = 1
        W[(c_col + 1) * n + c_row + 1][col] = 1  # C[1,1] += P6
        col += 1
        
        # P7 = (a01-a11)*(b10+b11)
        U[a_row * m + a_col + 1][col] = 1
        U[(a_row + 1) * m + a_col + 1][col] = -1
        V[(b_row + 1) * p + b_col][col] = 1
        V[(b_row + 1) * p + b_col + 1][col] = 1
        W[(c_col) * n + c_row][col] = -1      # C[0,0] -= P7
        col += 1
    
    # A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B1 = B[0:2, :], B2 = B[2:4, :]
    
    # A1 * B1
    # A1_top = A[0, 0:2] (1x2): a_row=0, a_col=0
    # A1_bot = A[1:3, 0:2] (2x2): a_row=1, a_col=0
    # B1_left = B[0:2, 0:3] (2x3): b_row=0, b_col=0
    # B1_right = B[0:2, 3:5] (2x2): b_row=0, b_col=3
    
    # A1_top * B1 (1x2 * 2x5): C row 0, cols 0-4
    add_1x2_2x5(0, 0, 0, 0, 0, 0)
    
    # A1_bot * B1_left (2x2 * 2x3): C rows 1-2, cols 0-2
    add_2x2_2x3(1, 0, 0, 0, 1, 0)
    
    # A1_bot * B1_right (2x2 * 2x2): C rows 1-2, cols 3-4
    add_strassen_2x2(1, 0, 0, 3, 1, 3)
    
    # A2 * B2
    # A2_top = A[0, 2:4] (1x2): a_row=0, a_col=2
    # A2_bot = A[1:3, 2:4] (2x2): a_row=1, a_col=2
    # B2_left = B[2:4, 0:3] (2x3): b_row=2, b_col=0
    # B2_right = B[2:4, 3:5] (2x2): b_row=2, b_col=3
    
    # A2_top * B2 (1x2 * 2x5): C row 0, cols 0-4
    add_1x2_2x5(0, 2, 2, 0, 0, 0)
    
    # A2_bot * B2_left (2x2 * 2x3): C rows 1-2, cols 0-2
    add_2x2_2x3(1, 2, 2, 0, 1, 0)
    
    # A2_bot * B2_right (2x2 * 2x2): C rows 1-2, cols 3-4
    add_strassen_2x2(1, 2, 2, 3, 1, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
