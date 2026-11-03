import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    
    # Calculate total rank
    # Row 0: 1x4 * 4x6 -> 3 blocks of (1x2 * 2x2), rank 4 each. 3 * (2 * 4) = 24
    # Rows 1-2: 2x4 * 4x6 -> 3 blocks of (2x2 * 2x2), rank 7 each. 3 * (2 * 7) = 42
    # Total rank = 66
    
    R = 66
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        # M1
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        # M3
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        # M6
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1

    def add_1x2_block(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        # M1
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        # M2
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        # M3
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        # M4
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1

    # Row 0: 1x4 * 4x6
    # Split A row 0 into two 1x2 blocks: cols 0-1, 2-3
    # Split B into three 4x2 blocks: cols 0-1, 2-3, 4-5
    # For each pair (A_block, B_block), we have 1x2 * 4x2
    # 1x2 * 4x2 is split into two 1x2 * 2x2 blocks (rows 0-1, 2-3 of B)
    
    a_row = 0
    a_splits = [(0, 0), (0, 2)] # a_r, a_c
    b_col_splits = [(0, 0), (0, 2), (0, 4)] # b_r, b_c (b_r is always 0)
    
    for a_r, a_c in a_splits:
        for b_r, b_c in b_col_splits:
            # A is 1x2 at (a_r, a_c)
            # B is 4x2 at (b_r, b_c)
            # Split B into two 2x2 blocks: rows 0-1, 2-3
            # B_block1: rows b_r to b_r+1, cols b_c to b_c+1
            # B_block2: rows b_r+2 to b_r+3, cols b_c to b_c+1
            
            # Wait, B is 4x2. Rows are 0-3.
            # The 1x2 A block multiplies with the 4x2 B block.
            # C is 1x2.
            # C_col = A[0,0] * B[0:2, 0:2] + A[0,1] * B[2:4, 0:2] ?? No.
            # A is 1x2. B is 4x2.
            # A = [a0, a1].
            # B = [[b00, b01], [b10, b11], [b20, b21], [b30, b31]].
            # C0 = a0*b00 + a1*b20.
            # C1 = a0*b01 + a1*b21.
            # This is NOT a standard matrix multiplication block.
            # The standard multiplication is $1 \times 4 \times 4 \times 2$.
            # My previous analysis assumed $1 \times 2 \times 2 \times 2$ blocks.
            # Let's re-verify the 1x4 * 4x2 decomposition.
            # A = [a0, a1, a2, a3].
            # C0 = a0 b00 + a1 b10 + a2 b20 + a3 b30.
            # C1 = a0 b01 + a1 b11 + a2 b21 + a3 b31.
            # Split A into [a0, a1] and [a2, a3].
            # Split B into B_top (rows 0-1) and B_bottom (rows 2-3).
            # C0 = a_top * b_top_col0 + a_bottom * b_bottom_col0.
            # C1 = a_top * b_top_col1 + a_bottom * b_bottom_col1.
            # a_top * b_top is 1x2 * 2x2 -> 1x2. Rank 4.
            # a_bottom * b_bottom is 1x2 * 2x2 -> 1x2. Rank 4.
            # Total rank 8.
            
            # So for each b_col_split, we add two 1x2 blocks.
            # Block 1: A[0, a_c:a_c+2] * B[0:2, b_c:b_c+2]
            # Block 2: A[0, a_c+2:a_c+4] * B[2:4, b_c:b_c+2]
            
            # Block 1
            add_1x2_block(a_r, a_c, b_r, b_c, 0, b_c)
            # Block 2
            add_1x2_block(a_r, a_c + 2, b_r + 2, b_c, 0, b_c)

    # Rows 1-2: 2x4 * 4x6
    # Split A into two 2x2 blocks: cols 0-1, 2-3
    # Split B into three 4x2 blocks: cols 0-1, 2-3, 4-5
    # For each pair, 2x2 * 4x2 is sum of two 2x2 * 2x2 blocks.
    
    a_row_start = 1
    a_splits = [(a_row_start, 0), (a_row_start, 2)]
    b_col_splits = [(0, 0), (0, 2), (0, 4)]
    
    for a_r, a_c in a_splits:
        for b_r, b_c in b_col_splits:
            # Block 1: A[a_r:a_r+2, a_c:a_c+2] * B[0:2, b_c:b_c+2]
            add_strassen_block(a_r, a_c, b_r, b_c, a_r, b_c)
            # Block 2: A[a_r:a_r+2, a_c:a_c+2] * B[2:4, b_c:b_c+2]
            add_strassen_block(a_r, a_c, b_r + 2, b_c, a_r, b_c)

    assert col == R, f"Expected {R} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
