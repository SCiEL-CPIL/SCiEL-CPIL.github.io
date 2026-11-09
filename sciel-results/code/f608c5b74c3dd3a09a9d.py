import numpy as np
import random
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Try to find a lower rank solution using Strassen-based approach
    # For 2x4x7, we can split into blocks and apply Strassen where beneficial
    
    # Approach: Split A (2x4) into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # Split B (4x7) into B_top (2x7, rows 0-1) and B_bot (2x7, rows 2-3)
    # C = A1 * B_top + A2 * B_bot
    
    # For each 2x2 * 2x7 multiplication, we can further split B into 2x2 + 2x2 + 2x1
    # and apply Strassen to the 2x2*2x2 parts
    
    # Let's build a decomposition of rank 50 using Strassen for 2x2*2x2 blocks
    # 2x2*2x2 via Strassen has rank 7
    # 2x2*2x1 via direct has rank 4
    
    # A1 (2x2) * B_top (2x7):
    #   Split B_top into B_top_L (2x2, cols 0-1), B_top_M (2x2, cols 2-3), B_top_R (2x1, col 4-6? No, 2x1 is 1 col)
    #   Actually B_top is 2x7, so split into 2x2, 2x2, 2x2, 2x1 (cols 0-1, 2-3, 4-5, 6)
    #   Strassen for 2x2*2x2: rank 7 each, 3 such blocks = 21
    #   Direct for 2x2*2x1: rank 4
    #   Total for A1*B_top: 21 + 4 = 25
    
    # A2 (2x2) * B_bot (2x7):
    #   Similarly: 21 + 4 = 25
    
    # Total rank: 25 + 25 = 50
    
    # Let's implement this properly
    U = [[0] * 50 for _ in range(n * m)]
    V = [[0] * 50 for _ in range(m * p)]
    W = [[0] * 50 for _ in range(p * n)]
    
    col = 0
    
    # Helper: Add Strassen decomposition for 2x2 * 2x2
    # A is 2x2, B is 2x2, C is 2x2
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add Strassen decomposition for 2x2*2x2.
        a_rows, a_cols: row and col indices of A in the full matrix
        b_rows, b_cols: row and col indices of B
        c_rows, c_cols: row and col indices of C
        """
        nonlocal col
        # A indices: (r1,c1), (r1,c2), (r2,c1), (r2,c2)
        a00 = a_rows[0] * m + a_cols[0]
        a01 = a_rows[0] * m + a_cols[1]
        a10 = a_rows[1] * m + a_cols[0]
        a11 = a_rows[1] * m + a_cols[1]
        
        # B indices: (r1,c1), (r1,c2), (r2,c1), (r2,c2)
        b00 = b_rows[0] * p + b_cols[0]
        b01 = b_rows[0] * p + b_cols[1]
        b10 = b_rows[1] * p + b_cols[0]
        b11 = b_rows[1] * p + b_cols[1]
        
        # C indices: (r1,c1), (r1,c2), (r2,c1), (r2,c2)
        c00 = c_rows[0] * n + c_cols[0]
        c01 = c_rows[0] * n + c_cols[1]
        c10 = c_rows[1] * n + c_cols[0]
        c11 = c_rows[1] * n + c_cols[1]
        
        # M1 = (A00+A11)*(B00+B11)
        U[a00][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M2 = (A10+A11)*B00
        U[a10][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1
        W[c11][col] = -1
        col += 1
        
        # M3 = A00*(B01-B11)
        U[a00][col] = 1
        V[b01][col] = 1
        V[b11][col] = -1
        W[c01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M4 = A11*(B10-B00)
        U[a11][col] = 1
        V[b10][col] = 1
        V[b00][col] = -1
        W[c00][col] = 1
        W[c10][col] = 1
        col += 1
        
        # M5 = (A00+A01)*B11
        U[a00][col] = 1
        U[a01][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1
        W[c01][col] = 1
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[a10][col] = 1
        U[a00][col] = -1
        V[b00][col] = 1
        V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[a01][col] = 1
        U[a11][col] = -1
        V[b10][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        W[c01][col] = -1
        col += 1
    
    def add_direct_2x2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col):
        """Add direct decomposition for 2x2 * 2x1.
        A is 2x2, B is 2x1, C is 2x1.
        """
        nonlocal col
        a00 = a_rows[0] * m + a_cols[0]
        a01 = a_rows[0] * m + a_cols[1]
        a10 = a_rows[1] * m + a_cols[0]
        a11 = a_rows[1] * m + a_cols[1]
        
        b0 = b_rows[0] * p + b_col
        b1 = b_rows[1] * p + b_col
        
        c0 = c_rows[0] * n + c_col
        c1 = c_rows[1] * n + c_col
        
        # C0 = A00*B0 + A01*B1
        U[a00][col] = 1
        V[b0][col] = 1
        W[c0][col] = 1
        col += 1
        
        U[a01][col] = 1
        V[b1][col] = 1
        W[c0][col] = 1
        col += 1
        
        # C1 = A10*B0 + A11*B1
        U[a10][col] = 1
        V[b0][col] = 1
        W[c1][col] = 1
        col += 1
        
        U[a11][col] = 1
        V[b1][col] = 1
        W[c1][col] = 1
        col += 1
    
    # A1 = A[0:2, 0:2], rows 0-1, cols 0-1
    # B_top = B[0:2, 0:7], rows 0-1, cols 0-6
    # C_top = C[0:2, 0:7], rows 0-1, cols 0-6
    
    # A1 * B_top_L (cols 0-1): Strassen
    add_strassen_2x2x2([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # A1 * B_top_M (cols 2-3): Strassen
    add_strassen_2x2x2([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # A1 * B_top_R (cols 4-5): Strassen
    add_strassen_2x2x2([0, 1], [0, 1], [0, 1], [4, 5], [0, 1], [4, 5])
    
    # A1 * B_top_last (col 6): Direct
    add_direct_2x2x1([0, 1], [0, 1], [0, 1], 6, [0, 1], 6)
    
    # A2 = A[0:2, 2:4], rows 0-1, cols 2-3
    # B_bot = B[2:4, 0:7], rows 2-3, cols 0-6
    # C_bot = C[0:2, 0:7], rows 0-1, cols 0-6
    
    # A2 * B_bot_L (cols 0-1): Strassen
    add_strassen_2x2x2([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # A2 * B_bot_M (cols 2-3): Strassen
    add_strassen_2x2x2([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # A2 * B_bot_R (cols 4-5): Strassen
    add_strassen_2x2x2([0, 1], [2, 3], [2, 3], [4, 5], [0, 1], [4, 5])
    
    # A2 * B_bot_last (col 6): Direct
    add_direct_2x2x1([0, 1], [2, 3], [2, 3], 6, [0, 1], 6)
    
    # Verify col is 50
    if col != 50:
        # Fallback to schoolbook if construction error
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
