"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a rank 22 decomposition.
    # 7 for A[:, 0:2] * B[0:2, 0:2] (Strassen)
    # 7 for A[:, 0:2] * B[0:2, 2:4] (Strassen)
    # 8 for A[:, 2:3] * B[2:3, 0:4] (Schoolbook)
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # a_row_offset, a_col_offset: top-left of A block (2x2)
    # b_row_offset, b_col_offset: top-left of B block (2x2)
    # c_row_offset, c_col_offset: top-left of C block (2x2)
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # Strassen multiplications
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Indices in A: (a_r, a_c) is (0,0) of block
        # A00 -> (a_r, a_c), A11 -> (a_r+1, a_c+1)
        # A10 -> (a_r+1, a_c), A01 -> (a_r, a_c+1)
        
        # Indices in B: (b_r, b_c) is (0,0) of block
        # B00 -> (b_r, b_c), B11 -> (b_r+1, b_c+1)
        # B01 -> (b_r, b_c+1), B10 -> (b_r+1, b_c)
        
        # Indices in C: (c_r, c_c) is (0,0) of block
        # C00 -> (c_r, c_c), C11 -> (c_r+1, c_c+1)
        # C01 -> (c_r, c_c+1), C10 -> (c_r+1, c_c)
        
        # M1
        U[a_r*m + a_c][col] = 1
        U[(a_r+1)*m + (a_c+1)][col] = 1
        V[b_r*p + b_c][col] = 1
        V[(b_r+1)*p + (b_c+1)][col] = 1
        W[c_c*n + c_r][col] = 1       # C00
        W[(c_c+1)*n + (c_r+1)][col] = 1 # C11
        col += 1
        
        # M2
        U[(a_r+1)*m + a_c][col] = 1
        U[(a_r+1)*m + (a_c+1)][col] = 1
        V[b_r*p + b_c][col] = 1
        W[c_c*n + (c_r+1)][col] = 1   # C10
        col += 1
        
        # M3
        U[a_r*m + a_c][col] = 1
        V[b_r*p + (b_c+1)][col] = 1
        V[(b_r+1)*p + (b_c+1)][col] = -1
        W[(c_c+1)*n + c_r][col] = 1   # C01
        col += 1
        
        # M4
        U[(a_r+1)*m + (a_c+1)][col] = 1
        V[(b_r+1)*p + b_c][col] = 1
        V[b_r*p + b_c][col] = -1
        W[c_c*n + c_r][col] = 1       # C00
        W[c_c*n + (c_r+1)][col] = 1   # C10
        col += 1
        
        # M5
        U[a_r*m + a_c][col] = 1
        U[a_r*m + (a_c+1)][col] = 1
        V[(b_r+1)*p + (b_c+1)][col] = 1
        W[c_c*n + c_r][col] = -1      # C00
        W[(c_c+1)*n + c_r][col] = 1   # C01
        col += 1
        
        # M6
        U[(a_r+1)*m + a_c][col] = 1
        U[a_r*m + a_c][col] = -1
        V[b_r*p + b_c][col] = 1
        V[b_r*p + (b_c+1)][col] = 1
        W[(c_c+1)*n + (c_r+1)][col] = 1 # C11
        col += 1
        
        # M7
        U[a_r*m + (a_c+1)][col] = 1
        U[(a_r+1)*m + (a_c+1)][col] = -1
        V[(b_r+1)*p + b_c][col] = 1
        V[(b_r+1)*p + (b_c+1)][col] = 1
        W[c_c*n + c_r][col] = 1       # C00
        W[(c_c+1)*n + (c_r+1)][col] = 1 # C11
        col += 1

    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    add_strassen(0, 0, 0, 2, 0, 2)
    
    # Block 3: A[:, 2:3] * B[2:3, 0:4] -> C[:, 0:4]
    # Schoolbook for 2x1 * 1x4
    for i in range(2):
        for k in range(4):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
