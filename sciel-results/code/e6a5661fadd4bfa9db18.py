import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will build a decomposition of rank 88.
    # 1. 2x4x8 part (using 2x4x4 rank 28 twice) -> 56
    # 2. 1x4x8 part (schoolbook) -> 32
    # Total 88.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # a_row, a_col, b_row, b_col, c_row, c_col are offsets
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        # 7 terms
        # M1
        U[a_r * m + a_c][start_col] = 1
        U[(a_r+1) * m + (a_c+1)][start_col] = 1
        V[b_r * p + b_c][start_col] = 1
        V[(b_r+1) * p + (b_c+1)][start_col] = 1
        W[c_r * n + c_c][start_col] = 1
        W[(c_r+1) * n + (c_c+1)][start_col] = 1
        col = start_col + 1
        
        # M2
        U[(a_r+1) * m + a_c][col] = 1
        U[(a_r+1) * m + (a_c+1)][col] = 1
        V[b_r * p + b_c][col] = 1
        W[(c_r+1) * n + c_c][col] = 1
        col += 1
        
        # M3
        U[a_r * m + a_c][col] = 1
        V[b_r * p + (b_c+1)][col] = -1
        V[(b_r+1) * p + (b_c+1)][col] = 1 # Wait, B12 - B22
        # B12 is (b_r, b_c+1), B22 is (b_r+1, b_c+1)
        W[c_r * n + (c_c+1)][col] = 1
        col += 1
        
        # M4
        U[(a_r+1) * m + (a_c+1)][col] = 1
        V[(b_r+1) * p + b_c][col] = 1
        V[b_r * p + b_c][col] = -1 # B21 - B11
        W[(c_r+1) * n + c_c][col] = 1
        col += 1
        
        # M5
        U[a_r * m + a_c][col] = 1
        U[a_r * m + (a_c+1)][col] = 1
        V[(b_r+1) * p + (b_c+1)][col] = 1
        W[c_r * n + c_c][col] = -1
        W[c_r * n + (c_c+1)][col] = 1
        col += 1
        
        # M6
        U[(a_r+1) * m + a_c][col] = 1
        U[a_r * m + a_c][col] = -1
        V[b_r * p + b_c][col] = 1
        V[b_r * p + (b_c+1)][col] = 1
        W[(c_r+1) * n + (c_c+1)][col] = 1
        col += 1
        
        # M7
        U[a_r * m + (a_c+1)][col] = 1
        U[(a_r+1) * m + (a_c+1)][col] = -1
        V[(b_r+1) * p + b_c][col] = 1
        V[(b_r+1) * p + (b_c+1)][col] = 1
        W[c_r * n + c_c][col] = 1
        W[(c_r+1) * n + (c_c+1)][col] = -1
        col += 1
        
        return col

    # 2x4x4 decomposition using two 2x2x4 (each 2 Strassen)
    # A is 2x4, B is 4x4, C is 2x4
    # Split B into B_top (2x4) and B_bot (2x4)
    # C = A_left B_top + A_right B_bot
    # A_left is 2x2, A_right is 2x2
    # A_left B_top is 2x2x4 (rank 14)
    # A_right B_bot is 2x2x4 (rank 14)
    
    def add_2x4x4(a_r, b_r, c_r, start_col):
        # A is at a_r, a_r+1 (rows), 0..3 (cols)
        # B is at b_r, b_r+1, b_r+2, b_r+3 (rows), 0..3 (cols)
        # C is at c_r, c_r+1 (rows), 0..3 (cols)
        
        # A_left (cols 0,1) * B_top (rows 0,1)
        # This is 2x2x2 + 2x2x2
        # B_top cols 0,1 -> C cols 0,1
        col = add_strassen(a_r, 0, b_r, 0, c_r, 0, start_col)
        # B_top cols 2,3 -> C cols 2,3
        col = add_strassen(a_r, 0, b_r, 2, c_r, 2, col)
        
        # A_right (cols 2,3) * B_bot (rows 2,3)
        # B_bot cols 0,1 -> C cols 0,1
        col = add_strassen(a_r, 2, b_r+2, 0, c_r, 0, col)
        # B_bot cols 2,3 -> C cols 2,3
        col = add_strassen(a_r, 2, b_r+2, 2, c_r, 2, col)
        
        return col

    # 2x4x8 decomposition
    # Split B into B_left (4x4) and B_right (4x4)
    # C_left (2x4) and C_right (2x4)
    
    # A is 2x4 (rows 0,1 of A in 3x4)
    # B is 4x8 (rows 0..3, cols 0..7)
    # C is 2x8 (rows 0,1 of C in 3x8)
    
    # A (rows 0,1) * B_left (cols 0-3) -> C_left (cols 0-3)
    col = add_2x4x4(0, 0, 0, col)
    
    # A (rows 0,1) * B_right (cols 4-7) -> C_right (cols 4-7)
    col = add_2x4x4(0, 0, 4, col) # Wait, c_r is row offset for C.
    # C_right is at rows 0,1, cols 4,5,6,7.
    # So c_r=0, c_c=4.
    # But add_2x4x4 takes c_r and assumes c_c=0?
    # My add_strassen takes c_c.
    # So I need to modify add_2x4x4 to take c_c.
    
    # Let's refine the helper functions to take all offsets.

    # Re-implementation with proper offsets
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        # ... (same as above) ...
        pass

    def add_2x4x4(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        # A is 2x4 at (a_r, a_c)
        # B is 4x4 at (b_r, b_c)
        # C is 2x4 at (c_r, c_c)
        
        # A_left (cols a_c, a_c+1) * B_top (rows b_r, b_r+1)
        # B_top cols b_c, b_c+1 -> C cols c_c, c_c+1
        col = add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col)
        # B_top cols b_c+2, b_c+3 -> C cols c_c+2, c_c+3
        col = add_strassen(a_r, a_c, b_r, b_c+2, c_r, c_c+2, col)
        
        # A_right (cols a_c+2, a_c+3) * B_bot (rows b_r+2, b_r+3)
        # B_bot cols b_c, b_c+1 -> C cols c_c, c_c+1
        col = add_strassen(a_r, a_c+2, b_r+2, b_c, c_r, c_c, col)
        # B_bot cols b_c+2, b_c+3 -> C cols c_c+2, c_c+3
        col = add_strassen(a_r, a_c+2, b_r+2, b_c+2, c_r, c_c+2, col)
        
        return col

    # 2x4x8
    # A is 2x4 at (0,0)
    # B is 4x8 at (0,0)
    # C is 2x8 at (0,0)
    
    # Left half of B (cols 0-3)
    col = add_2x4x4(0, 0, 0, 0, 0, 0, col)
    
    # Right half of B (cols 4-7)
    col = add_2x4x4(0, 0, 0, 4, 0, 4, col)
    
    # 1x4x8 schoolbook
    # A is 1x4 at (2,0)
    # B is 4x8 at (0,0)
    # C is 1x8 at (2,0)
    
    for j in range(4):
        for k in range(8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
