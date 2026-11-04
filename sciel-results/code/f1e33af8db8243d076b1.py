import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # Split A (2x3) into A_L (2x2) and A_R (2x1)
    # Split B (3x4) into B_T (2x4) and B_B (1x4)
    # C = A_L * B_T + A_R * B_B
    
    # Rank of A_L * B_T (2x2 * 2x4):
    # Split B_T into B_T1 (2x2) and B_T2 (2x2)
    # A_L * B_T1 uses Strassen (rank 7)
    # A_L * B_T2 uses Strassen (rank 7)
    # Total 14
    
    # Rank of A_R * B_B (2x1 * 1x4):
    # This is 8 scalar multiplications (rank 8)
    # Total 14 + 8 = 22
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_L (cols 0,1) * B_T (rows 0,1)
    # A_L is 2x2, B_T is 2x4 (split into two 2x2)
    
    # Block 1: A_L * B_T1 (cols 0,1 of B)
    # Strassen for 2x2 * 2x2
    # Indices:
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1]
    # B_T1: B[0,0], B[0,1], B[1,0], B[1,1]
    # C: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # M1 = A00 * (B00 + B11)
    U[0][col] = 1 # A00
    V[0][col] = 1 # B00
    V[3][col] = 1 # B11
    W[0][col] = 1 # C00
    W[1][col] = 1 # C01? No, W index is k*n+i.
    # C00 is i=0, k=0 -> W[0]
    # C01 is i=0, k=1 -> W[1]
    # C10 is i=1, k=0 -> W[2]
    # C11 is i=1, k=1 -> W[3]
    
    # Let's write a helper to add a Strassen block
    def add_strassen(U, V, W, col, a_row, a_col, b_row, b_col, c_row, c_col):
        # a_row, a_col: start indices for A (2x2)
        # b_row, b_col: start indices for B (2x2)
        # c_row, c_col: start indices for C (2x2)
        
        # A indices:
        a00 = a_row * m + a_col
        a01 = a_row * m + a_col + 1
        a10 = (a_row + 1) * m + a_col
        a11 = (a_row + 1) * m + a_col + 1
        
        # B indices:
        b00 = b_row * p + b_col
        b01 = b_row * p + b_col + 1
        b10 = (b_row + 1) * p + b_col
        b11 = (b_row + 1) * p + b_col + 1
        
        # C indices (W index: k*n + i)
        # C[i, k]
        c00 = b_col * n + a_row
        c01 = (b_col + 1) * n + a_row
        c10 = b_col * n + (a_row + 1)
        c11 = (b_col + 1) * n + (a_row + 1)
        
        # M1 = A00 * (B00 + B11)
        U[a00][col] = 1
        V[b00][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        W[c10][col] = 1 # Wait, C10 is i=1, k=0.
        # C00 = A00(B00+B11) - A00B11? No.
        # Standard Strassen:
        # M1 = A00(B00+B11)
        # M2 = A11(B00+B11)
        # M3 = (A00+A11)B11
        # M4 = A01(B10-B11)
        # M5 = (A10-A00)(B00+B01)
        # M6 = (A00+A01)B10
        # M7 = (A10+A11)B01
        #
        # C00 = M1 - M3 + M5
        # C01 = M4 + M6
        # C10 = M2 - M5 + M7
        # C11 = M1 - M2 + M3
        
        # We need to add these to the existing U, V, W.
        # Since we are constructing from scratch, we can just add the terms.
        
        # M1
        U[a00][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c10][col] += -1 # C10 = M2 - M5 + M7, M1 contributes to C00 and C11?
        # Let's trace:
        # C00 = M1 - M3 + M5
        # C11 = M1 - M2 + M3
        # So M1 contributes to C00 and C11.
        W[c00][col] += 1
        W[c11][col] += 1
        
        # M2 = A11(B00+B11)
        # C10 = M2 - M5 + M7
        # C11 = M1 - M2 + M3
        U[a11][col+1] += 1
        V[b00][col+1] += 1
        V[b11][col+1] += 1
        W[c10][col+1] += 1
        W[c11][col+1] += -1
        
        # M3 = (A00+A11)B11
        # C00 = M1 - M3 + M5
        # C11 = M1 - M2 + M3
        U[a00][col+2] += 1
        U[a11][col+2] += 1
        V[b11][col+2] += 1
        W[c00][col+2] += -1
        W[c11][col+2] += 1
        
        # M4 = A01(B10-B11)
        # C01 = M4 + M6
        U[a01][col+3] += 1
        V[b10][col+3] += 1
        V[b11][col+3] += -1
        W[c01][col+3] += 1
        
        # M5 = (A10-A00)(B00+B01)
        # C00 = M1 - M3 + M5
        # C10 = M2 - M5 + M7
        U[a10][col+4] += 1
        U[a00][col+4] += -1
        V[b00][col+4] += 1
        V[b01][col+4] += 1
        W[c00][col+4] += 1
        W[c10][col+4] += -1
        
        # M6 = (A00+A01)B10
        # C01 = M4 + M6
        U[a00][col+5] += 1
        U[a01][col+5] += 1
        V[b10][col+5] += 1
        W[c01][col+5] += 1
        
        # M7 = (A10+A11)B01
        # C10 = M2 - M5 + M7
        U[a10][col+6] += 1
        U[a11][col+6] += 1
        V[b01][col+6] += 1
        W[c10][col+6] += 1
        
        return col + 7

    # Add Strassen block for A_L (rows 0,1; cols 0,1) * B_T1 (rows 0,1; cols 0,1)
    col = add_strassen(U, V, W, col, 0, 0, 0, 0, 0, 0)
    
    # Add Strassen block for A_L (rows 0,1; cols 0,1) * B_T2 (rows 0,1; cols 2,3)
    col = add_strassen(U, V, W, col, 0, 0, 0, 2, 0, 2)
    
    # Part 2: A_R (cols 2) * B_B (rows 2)
    # A_R is 2x1, B_B is 1x4
    # C[i, k] = A[i, 2] * B[2, k]
    # This is 8 terms.
    for i in range(2):
        for k in range(4):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
