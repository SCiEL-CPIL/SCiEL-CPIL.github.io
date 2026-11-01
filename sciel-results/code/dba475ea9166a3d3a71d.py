"""Optimized rank-14 decomposition for 2x2x4 matrix multiplication."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    
    # U: 4x14, V: 8x14, W: 8x14
    # Column layout: c=0..6 are Strassen for A*B_left, c=7..13 for A*B_right
    
    # Strassen products for first 2x2 block (columns 0-6)
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A00+A11)B10
    # P2 = A00(B00-B10)
    # P3 = A11(B11-B01)
    # P4 = (A10+A11)B01
    # P5 = (A00-A10)(B00+B10)
    # P6 = (A01-A11)(B01+B11)
    
    # For the second 2x2 block (columns 7-13), we reuse the same U factors
    # but with different V factors that mix columns from both B blocks
    
    U = [
        [0,0,0,0,0,0,0, 0,0,0,0,0,0,0],  # row 0: A00
        [0,0,0,0,0,0,0, 0,0,0,0,0,0,0],  # row 1: A01
        [0,0,0,0,0,0,0, 0,0,0,0,0,0,0],  # row 2: A10
        [0,0,0,0,0,0,0, 0,0,0,0,0,0,0],  # row 3: A11
    ]
    
    # Set U factors based on Strassen linear combinations
    # Column 0: A00 + A11
    U[0][0] = 1; U[3][0] = 1
    # Column 1: A00 + A11
    U[0][1] = 1; U[3][1] = 1
    # Column 2: A00
    U[0][2] = 1
    # Column 3: A11
    U[3][3] = 1
    # Column 4: A10 + A11
    U[2][4] = 1; U[3][4] = 1
    # Column 5: A00 - A10
    U[0][5] = 1; U[2][5] = -1
    # Column 6: A01 - A11
    U[1][6] = 1; U[3][6] = -1
    
    # Columns 7-13 reuse the same U patterns as columns 0-6
    for c in range(7):
        for i in range(4):
            U[i][c+7] = U[i][c]
    
    # V: 8x14
    # Rows: B00(0), B01(1), B02(2), B03(3), B10(4), B11(5), B12(6), B13(7)
    V = [[0]*rank for _ in range(8)]
    
    # First block (columns 0-6): standard Strassen V factors
    # P0 = (A00+A11)(B00+B11)
    V[0][0] = 1; V[5][0] = 1
    # P1 = (A00+A11)B10
    V[4][1] = 1
    # P2 = A00(B00-B10)
    V[0][2] = 1; V[4][2] = -1
    # P3 = A11(B11-B01)
    V[1][3] = -1; V[5][3] = 1
    # P4 = (A10+A11)B01
    V[1][4] = 1
    # P5 = (A00-A10)(B00+B10)
    V[0][5] = 1; V[4][5] = 1
    # P6 = (A01-A11)(B01+B11)
    V[1][6] = 1; V[5][6] = 1
    
    # Second block (columns 7-13): shifted Strassen using right 2x2 block
    # The right block is B[:, 2:4], so rows are B02(2), B03(3), B12(6), B13(7)
    # P7 = (A00+A11)(B02+B12)
    V[2][7] = 1; V[6][7] = 1
    # P8 = (A00+A11)B12
    V[6][8] = 1
    # P9 = A00(B02-B12)
    V[2][9] = 1; V[6][9] = -1
    # P10 = A11(B13-B03)
    V[3][10] = -1; V[7][10] = 1
    # P11 = (A10+A11)B03
    V[3][11] = 1
    # P12 = (A00-A10)(B02+B12)
    V[2][12] = 1; V[6][12] = 1
    # P13 = (A01-A11)(B03+B13)
    V[3][13] = 1; V[7][13] = 1
    
    # W: 8x14
    # Rows: C00(0), C10(1), C01(2), C11(3), C02(4), C12(5), C03(6), C13(7)
    W = [[0]*rank for _ in range(8)]
    
    # First block contribution (columns 0-6) to C[:, 0:2]
    # C00 = P0 + P2 + P3 - P5 - P6
    W[0][0] = 1; W[0][2] = 1; W[0][3] = 1; W[0][5] = -1; W[0][6] = -1
    # C10 = P1 + P5
    W[1][1] = 1; W[1][5] = 1
    # C01 = P4 + P6
    W[2][4] = 1; W[2][6] = 1
    # C11 = P0 + P1 - P3 - P4
    W[3][0] = 1; W[3][1] = 1; W[3][3] = -1; W[3][4] = -1
    
    # Second block contribution (columns 7-13) to C[:, 2:4]
    # C02 = P7 + P9 + P10 - P12 - P13
    W[4][7] = 1; W[4][9] = 1; W[4][10] = 1; W[4][12] = -1; W[4][13] = -1
    # C12 = P8 + P12
    W[5][8] = 1; W[5][12] = 1
    # C03 = P11 + P13
    W[6][11] = 1; W[6][13] = 1
    # C13 = P7 + P8 - P10 - P11
    W[7][7] = 1; W[7][8] = 1; W[7][10] = -1; W[7][11] = -1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
