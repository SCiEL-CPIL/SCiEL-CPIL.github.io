import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition with Strassen's algorithm for 2x2x2 blocks.
    # A is 2x4, B is 4x4, C is 2x4.
    # Split A into A_L (2x2) and A_R (2x2).
    # Split B into 4 2x2 blocks: B_00, B_01, B_10, B_11.
    # C_L = A_L B_00 + A_R B_10
    # C_R = A_L B_01 + A_R B_11
    # Each 2x2x2 multiplication has rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B10)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen(a_col_off, b_row_off, c_col_off):
        nonlocal col
        # A indices: row r (0,1), col a_col_off + c (0,1)
        # B indices: row b_row_off + r (0,1), col c (0,1)
        # C indices: row r (0,1), col c_col_off + c (0,1)
        
        # M1
        # A: (0,0)+ (1,1) -> (0, a_col_off), (1, a_col_off+1)
        # B: (0,0)+ (1,1) -> (b_row_off, 0), (b_row_off+1, 1)
        # C: (0,0), (1,1) -> (0, c_col_off), (1, c_col_off+1)
        # W index for C(r, c) is c*n + r.
        
        # We define the linear forms for U, V, W for each M_k
        # U entry at (r, a_col_off+c) gets coeff
        # V entry at (b_row_off+r, c) gets coeff
        # W entry at (c_col_off+c, r) gets coeff
        
        # M1
        U[0*m + a_col_off][col] += 1
        U[1*m + a_col_off+1][col] += 1
        V[b_row_off*p + 0][col] += 1
        V[(b_row_off+1)*p + 1][col] += 1
        # C00 gets +1, C11 gets +1
        W[0*n + 0][col] += 1 # c=0, r=0 -> 0*2+0=0
        W[1*n + 1][col] += 1 # c=1, r=1 -> 1*2+1=3
        col += 1
        
        # M2
        # A: (1,0)+(1,1)
        # B: (0,0)
        U[1*m + a_col_off][col] += 1
        U[1*m + a_col_off+1][col] += 1
        V[b_row_off*p + 0][col] += 1
        # C10 gets +1
        W[0*n + 1][col] += 1 # c=0, r=1 -> 1
        col += 1
        
        # M3
        # A: (0,0)
        # B: (1,0)-(1,1)
        U[0*m + a_col_off][col] += 1
        V[(b_row_off+1)*p + 0][col] += 1
        V[(b_row_off+1)*p + 1][col] += -1
        # C01 gets +1
        W[1*n + 0][col] += 1 # c=1, r=0 -> 2
        col += 1
        
        # M4
        # A: (1,1)
        # B: (0,1)-(0,0)
        U[1*m + a_col_off+1][col] += 1
        V[b_row_off*p + 1][col] += 1
        V[b_row_off*p + 0][col] += -1
        # C00 gets +1, C10 gets +1
        W[0*n + 0][col] += 1
        W[0*n + 1][col] += 1
        col += 1
        
        # M5
        # A: (0,0)+(1,0)
        # B: (1,1)
        U[0*m + a_col_off][col] += 1
        U[1*m + a_col_off][col] += 1
        V[(b_row_off+1)*p + 1][col] += 1
        # C00 gets -1, C01 gets +1
        W[0*n + 0][col] += -1
        W[1*n + 0][col] += 1
        col += 1
        
        # M6
        # A: (1,1)-(0,0)
        # B: (0,0)+(1,0)
        U[1*m + a_col_off+1][col] += 1
        U[0*m + a_col_off][col] += -1
        V[b_row_off*p + 0][col] += 1
        V[(b_row_off+1)*p + 0][col] += 1
        # C11 gets +1
        W[1*n + 1][col] += 1
        col += 1
        
        # M7
        # A: (0,1)-(1,1)
        # B: (1,1)+(1,0)
        U[0*m + a_col_off+1][col] += 1
        U[1*m + a_col_off+1][col] += -1
        V[(b_row_off+1)*p + 1][col] += 1
        V[(b_row_off+1)*p + 0][col] += 1
        # C00 gets +1, C11 gets +1? No, C11 = M1 - M2 + M3 + M6. M7 is in C00.
        # C00 = M1 + M4 - M5 + M7
        W[0*n + 0][col] += 1
        col += 1

    # Block 1: A_L (cols 0,1), B_00 (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    add_strassen(0, 0, 0)
    
    # Block 2: A_R (cols 2,3), B_10 (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    add_strassen(2, 2, 0)
    
    # Block 3: A_L (cols 0,1), B_01 (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    add_strassen(0, 0, 2)
    
    # Block 4: A_R (cols 2,3), B_11 (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    add_strassen(2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
