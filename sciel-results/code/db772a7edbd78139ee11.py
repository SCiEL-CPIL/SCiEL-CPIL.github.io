"""Exact rank-minimal tensor decomposition for 2x3 @ 3x8 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decomposition strategy:
    # Split B (3x8) into two 3x4 blocks (cols 0-3 and 4-7).
    # For each 3x4 block, split A (2x3) into A_top (2x2) and A_bottom (2x1),
    # and B_block (3x4) into B_top (2x4) and B_bottom (1x4).
    # C_block = A_top @ B_top + A_bottom @ B_bottom
    # A_top @ B_top: split B_top into two 2x2 blocks, use Strassen (rank 7) for each.
    # A_bottom @ B_bottom: 2x1 @ 1x4 = outer product, rank 8.
    # Total per block: 7+7+8 = 22. Total: 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def _add_strassen_2x2(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        # M1 = (A00+A11)*(B00+B11)
        U[a_row * m + a_col][col] = 1
        U[(a_row+1) * m + (a_col+1)][col] = 1
        V[b_row * p + b_col][col] = 1
        V[(b_row+1) * p + (b_col+1)][col] = 1
        W[(c_col + 0) * n + (c_row + 0)][col] = 1
        W[(c_col + 1) * n + (c_row + 1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)*B00
        U[(a_row+1) * m + a_col][col] = 1
        U[(a_row+1) * m + (a_col+1)][col] = 1
        V[b_row * p + b_col][col] = 1
        W[(c_col + 0) * n + (c_row + 1)][col] = 1
        W[(c_col + 1) * n + (c_row + 1)][col] = -1
        col += 1
        
        # M3 = A00*(B01-B11)
        U[a_row * m + a_col][col] = 1
        V[b_row * p + (b_col+1)][col] = 1
        V[(b_row+1) * p + (b_col+1)][col] = -1
        W[(c_col + 1) * n + (c_row + 0)][col] = 1
        W[(c_col + 1) * n + (c_row + 1)][col] = 1
        col += 1
        
        # M4 = A11*(B10-B00)
        U[(a_row+1) * m + (a_col+1)][col] = 1
        V[(b_row+1) * p + b_col][col] = 1
        V[b_row * p + b_col][col] = -1
        W[(c_col + 0) * n + (c_row + 0)][col] = 1
        W[(c_col + 0) * n + (c_row + 1)][col] = 1
        col += 1
        
        # M5 = (A00+A01)*B11
        U[a_row * m + a_col][col] = 1
        U[a_row * m + (a_col+1)][col] = 1
        V[(b_row+1) * p + (b_col+1)][col] = 1
        W[(c_col + 0) * n + (c_row + 0)][col] = -1
        W[(c_col + 1) * n + (c_row + 0)][col] = 1
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[(a_row+1) * m + a_col][col] = 1
        U[a_row * m + a_col][col] = -1
        V[b_row * p + b_col][col] = 1
        V[b_row * p + (b_col+1)][col] = 1
        W[(c_col + 1) * n + (c_row + 1)][col] = 1
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[a_row * m + (a_col+1)][col] = 1
        U[(a_row+1) * m + (a_col+1)][col] = -1
        V[(b_row+1) * p + b_col][col] = 1
        V[(b_row+1) * p + (b_col+1)][col] = 1
        W[(c_col + 0) * n + (c_row + 0)][col] = 1
        col += 1
    
    def _add_outer_2x1_1x4(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        for i in range(2):
            for j in range(4):
                U[(a_row + i) * m + a_col][col] = 1
                V[(b_row + 0) * p + (b_col + j)][col] = 1
                W[(c_col + j) * n + (c_row + i)][col] = 1
                col += 1
    
    # Process two 3x4 blocks
    for blk in range(2):
        bcs = 4 * blk  # B column start
        ccs = 4 * blk  # C column start
        
        # A_top = rows 0,1 of A, cols 0,1
        # B_top = rows 0,1 of B, cols bcs..bcs+3
        # Split B_top into two 2x2: cols bcs,bcs+1 and bcs+2,bcs+3
        _add_strassen_2x2(0, 0, 0, bcs,     0, ccs)
        _add_strassen_2x2(0, 0, 0, bcs + 2, 0, ccs + 2)
        
        # A_bottom = rows 0,1 of A, col 2
        # B_bottom = row 2 of B, cols bcs..bcs+3
        _add_outer_2x1_1x4(0, 2, 2, bcs, 0, ccs)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
