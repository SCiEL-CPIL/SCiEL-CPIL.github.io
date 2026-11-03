import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # Rank 36 decomposition
    # Part 1: 2x4x4 (Rank 28) using 4x Strassen(2x2x2)
    # Part 2: 2x4x1 (Rank 8) using 2x Standard(2x2x1)
    
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)] # 8x36
    V = [[0.0] * rank for _ in range(m * p)] # 20x36
    W = [[0.0] * rank for _ in range(p * n)] # 10x36
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # a_row_start, a_col_start: start index of A block (2x2) in U
    # b_row_start, b_col_start: start index of B block (2x2) in V
    # c_row_start, c_col_start: start index of C block (2x2) in W
    # Note: W index is k*n + i.
    # C is 2x2.
    # C_{00} is at (0*2+0, 0*2+0) -> (0,0)
    # C_{01} is at (0*2+0, 0*2+1) -> (0,1)
    # C_{10} is at (1*2+0, 0*2+0) -> (2,0)
    # C_{11} is at (1*2+0, 0*2+1) -> (2,1)
    
    def add_strassen_222(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # A indices in global U:
        # A_{00} at (a_r, a_c)
        # A_{01} at (a_r, a_c+1)
        # A_{10} at (a_r+1, a_c)
        # A_{11} at (a_r+1, a_c+1)
        
        # B indices in global V:
        # B_{00} at (b_r, b_c)
        # B_{01} at (b_r, b_c+1)
        # B_{10} at (b_r+1, b_c)
        # B_{11} at (b_r+1, b_c+1)
        
        # W indices in global W:
        # W index is k*n + i for C_{ki}
        # C is 2x2.
        # C_{00} -> k=0, i=0 -> idx 0
        # C_{01} -> k=0, i=1 -> idx 1
        # C_{10} -> k=1, i=0 -> idx 2
        # C_{11} -> k=1, i=1 -> idx 3
        # Global W row = c_r + local_row
        # Global W col = c_c + local_col (but W is 1D index k*n+i)
        
        # Strassen M1 = (A00+A11)B00
        U[a_r][col] += 1; U[a_r+1][col] += 1
        V[b_r][col] += 1
        W[c_r + 0][col] += 1  # C00
        W[c_r + 2][col] += 1  # C10 (C11 is c_r+2+1? No, C11 is k=1,i=1 -> 1*2+1=3. C10 is k=1,i=0 -> 1*2+0=2)
        # Wait, W[c_r + 0] is C00. W[c_r + 2] is C10.
        # C11 is c_r + 3.
        col += 1
        
        # M2 = (A10+A11)B01
        U[a_r+1][col] += 1; U[a_r+1][col] += 1 # A10+A11
        # Correction: A10 is (a_r+1, a_c), A11 is (a_r+1, a_c+1)
        U[a_r+1][col] = 1
        U[a_r+1][col] = 1 # This is wrong, need to add to different columns if same row? No, U is matrix.
        # U[row][col]
        # A10 is at row a_r+1, col a_c.
        # A11 is at row a_r+1, col a_c+1.
        # So U[a_r+1][col] += 1 for A10? No, U is the factor matrix.
        # U[i*m+j, c].
        # So U[row][col] corresponds to A element.
        # We need U[a_r+1, a_c][col] = 1 and U[a_r+1, a_c+1][col] = 1?
        # No, U is (n*m, rank).
        # So U is a 2D array.
        # U[row][col] is the coefficient.
        
        # Let's restart the helper logic to be precise.
        
    # Re-implementing carefully
    
    # We will just build the matrices directly using loops for the 4 Strassen blocks and 2 standard blocks.
    
    # Block 1: A0 (rows 0-1, cols 0-1) * B00 (rows 0-1, cols 0-1) -> C00 (rows 0-1, cols 0-1)
    # A0 is A[0:2, 0:2]
    # B00 is B[0:2, 0:2]
    # C00 is C[0:2, 0:2]
    
    # Indices:
    # A00: (0,0) -> U[0]
    # A01: (0,1) -> U[1]
    # A10: (1,0) -> U[2]
    # A11: (1,1) -> U[3]
    
    # B00: (0,0) -> V[0]
    # B01: (0,1) -> V[1]
    # B10: (1,0) -> V[2]
    # B11: (1,1) -> V[3]
    
    # C00: (0,0) -> W[0]
    # C01: (0,1) -> W[1]
    # C10: (1,0) -> W[2]
    # C11: (1,1) -> W[3]
    
    def add_strassen_block(a_base, b_base, c_base):
        nonlocal col
        # a_base: index of A00 in U
        # b_base: index of B00 in V
        # c_base: index of C00 in W
        
        # M1
        U[a_base][col] += 1
        U[a_base+3][col] += 1
        V[b_base][col] += 1
        W[c_base][col] += 1
        W[c_base+2][col] += 1
        col += 1
        
        # M2
        U[a_base+2][col] += 1
        U[a_base+3][col] += 1
        V[b_base+1][col] += 1
        W[c_base+1][col] += 1
        W[c_base+3][col] += 1
        col += 1
        
        # M3
        U[a_base][col] += 1
        V[b_base+2][col] += 1
        V[b_base+3][col] += 1
        W[c_base+1][col] += 1
        W[c_base+3][col] += 1
        col += 1
        
        # M4
        U[a_base+3][col] += 1
        V[b_base+1][col] += 1
        V[b_base+2][col] += -1
        W[c_base][col] += -1
        W[c_base+1][col] += 1
        col += 1
        
        # M5
        U[a_base][col] += 1
        U[a_base+1][col] += 1
        V[b_base+3][col] += 1
        W[c_base+2][col] += 1
        W[c_base+3][col] += 1
        col += 1
        
        # M6
        U[a_base+2][col] += 1
        U[a_base][col] += -1
        V[b_base][col] += 1
        V[b_base+1][col] += 1
        W[c_base][col] += 1
        col += 1
        
        # M7
        U[a_base+1][col] += 1
        U[a_base+3][col] += -1
        V[b_base+2][col] += 1
        V[b_base+3][col] += 1
        W[c_base][col] += -1
        W[c_base+3][col] += 1
        col += 1

    # Block 1: A0 * B00 -> C00
    # A0 base: 0
    # B00 base: 0
    # C00 base: 0
    add_strassen_block(0, 0, 0)
    
    # Block 2: A0 * B01 -> C01
    # A0 base: 0
    # B01 base: 2 (B01 is cols 2-3 of B, rows 0-1. B00 is cols 0-1. So B01 is V[2], V[3], V[4], V[5]?)
    # V index: j*p + k.
    # B is 4x5.
    # B00: j=0,1; k=0,1. V[0], V[1], V[2], V[3].
    # B01: j=0,1; k=2,3. V[0*5+2]=V[2], V[0*5+3]=V[3], V[1*5+2]=V[7], V[1*5+3]=V[8].
    # Wait, V index is j*p + k.
    # B00: (0,0)->0, (0,1)->1, (1,0)->5, (1,1)->6.
    # B01: (0,2)->2, (0,3)->3, (1,2)->7, (1,3)->8.
    # B10: (2,0)->10, (2,1)->11, (3,0)->15, (3,1)->16.
    # B11: (2,2)->12, (2,3)->13, (3,2)->17, (3,3)->18.
    
    # My previous V indexing was wrong.
    # V is (m*p, rank) = (20, rank).
    # V[j*p + k][c].
    
    # Let's re-map indices.
    # A (2x4). U[i*m+j][c].
    # A00: i=0, j=0 -> U[0]
    # A01: i=0, j=1 -> U[1]
    # A10: i=1, j=0 -> U[4]
    # A11: i=1, j=1 -> U[5]
    # A12: i=1, j=2 -> U[6]
    # A13: i=1, j=3 -> U[7]
    
    # B (4x5). V[j*p+k][c].
    # B00: j=0, k=0 -> V[0]
    # B01: j=0, k=1 -> V[1]
    # B10: j=1, k=0 -> V[5]
    # B11: j=1, k=1 -> V[6]
    # B02: j=0, k=2 -> V[2]
    # B03: j=0, k=3 -> V[3]
    # B12: j=1, k=2 -> V[7]
    # B13: j=1, k=3 -> V[8]
    # B20: j=2, k=0 -> V[10]
    # B21: j=2, k=1 -> V[11]
    # B30: j=3, k=0 -> V[15]
    # B31: j=3, k=1 -> V[16]
    # B22: j=2, k=2 -> V[12]
    # B23: j=2, k=3 -> V[13]
    # B32: j=3, k=2 -> V[17]
    # B33: j=3, k=3 -> V[18]
    # B04: j=0, k=4 -> V[4]
    # B14: j=1, k=4 -> V[9]
    # B24: j=2, k=4 -> V[14]
    # B34: j=3, k=4 -> V[19]
    
    # C (2x5). W[k*n+i][c].
    # C00: k=0, i=0 -> W[0]
    # C01: k=0, i=1 -> W[1]
    # C10: k=1, i=0 -> W[2]
    # C11: k=1, i=1 -> W[3]
    # C02: k=2, i=0 -> W[4]
    # C03: k=2, i=1 -> W[5]
    # C12: k=3, i=0 -> W[6]
    # C13: k=3, i=1 -> W[7]
    # C04: k=4, i=0 -> W[8]
    # C14: k=4, i=1 -> W[9]
    
    # Block 1: A0 (U[0],U[1],U[4],U[5]) * B00 (V[0],V[1],V[5],V[6]) -> C00 (W[0],W[1],W[2],W[3])
    # A00=U[0], A01=U[1], A10=U[4], A11=U[5]
    # B00=V[0], B01=V[1], B10=V[5], B11=V[6]
    
    # Block 2: A0 * B01 (V[2],V[3],V[7],V[8]) -> C01 (W[4],W[5],W[6],W[7])
    # B02=V[2], B03=V[3], B12=V[7], B13=V[8]
    
    # Block 3: A1 (U[2],U[3],U[6],U[7]) * B10 (V[10],V[11],V[15],V[16]) -> C10 (W[0],W[1],W[2],W[3])?
    # No, C = A0 B00 + A1 B10 is C00?
    # C00 is cols 0-1.
    # A0 B00 contributes to C00.
    # A1 B10 contributes to C00.
    # So C00 = A0 B00 + A1 B10.
    # We need to add both Strassen blocks to the same W indices.
    
    # Block 4: A1 * B11 (V[12],V[13],V[17],V[18]) -> C01 (W[4],W[5],W[6],W[7])
    # C01 = A0 B01 + A1 B11.
    
    # Part 2: C02 (W[8],W[9]) = A0 B04 + A1 B14?
    # B04 is j=0,k=4 -> V[4]. B14 is j=1,k=4 -> V[9].
    # B24 is j=2,k=4 -> V[14]. B34 is j=3,k=4 -> V[19].
    # C02 = A0 B04 + A1 B14 + A2 B24 + A3 B34?
    # A is 2x4. A0 is cols 0-1. A1 is cols 2-3.
    # A = [A0, A1].
    # B = [B0; B1; B2; B3] where Bj is 1x5? No.
    # B is 4x5.
    # B0 is rows 0-1, cols 0-4.
    # B1 is rows 2-3, cols 0-4.
    # C = A0 B0 + A1 B1.
    # C02 (col 4 of C) = A0 B0[:,4] + A1 B1[:,4].
    # B0[:,4] is V[4] (j=0,k=4) and V[9] (j=1,k=4).
    # B1[:,4] is V[14] (j=2,k=4) and V[19] (j=3,k=4).
    
    # So C02 = A0 * [V[4], V[9]] + A1 * [V[14], V[19]].
    # This is 2x2x1 multiplication twice.
    
    # Let's implement.
    
    # Reset
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen_222(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col
        # M1
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        # M2
        U[a10][col] += 1; U[a11][col] += 1
        V[b01][col] += 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        # M3
        U[a00][col] += 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        # M4
        U[a11][col] += 1
        V[b01][col] += 1; V[b10][col] += -1
        W[c00][col] += -1; W[c01][col] += 1
        col += 1
        # M5
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c10][col] += 1; W[c11][col] += 1
        col += 1
        # M6
        U[a10][col] += 1; U[a00][col] += -1
        V[b00][col] += 1; V[b01][col] += 1
        W[c00][col] += 1
        col += 1
        # M7
        U[a01][col] += 1; U[a11][col] += -1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += -1; W[c11][col] += 1
        col += 1

    # Block 1: A0 * B00 -> C00
    add_strassen_222(0, 1, 4, 5, 0, 1, 5, 6, 0, 1, 2, 3)
    # Block 3: A1 * B10 -> C00
    add_strassen_222(2, 3, 6, 7, 10, 11, 15, 16, 0, 1, 2, 3)
    # Block 2: A0 * B01 -> C01
    add_strassen_222(0, 1, 4, 5, 2, 3, 7, 8, 4, 5, 6, 7)
    # Block 4: A1 * B11 -> C01
    add_strassen_222(2, 3, 6, 7, 12, 13, 17, 18, 4, 5, 6, 7)
    
    # Part 2: C02 (W[8], W[9])
    # A0 * B04 (V[4], V[9])
    # A00=U[0], A01=U[1], A10=U[4], A11=U[5]
    # B04=V[4], B14=V[9]
    # C02 = A00*V[4] + A01*V[9] (for row 0)
    #       A10*V[4] + A11*V[9] (for row 1)
    
    # Term 1: A00 * V[4]
    U[0][col] += 1; V[4][col] += 1; W[8][col] += 1
    col += 1
    # Term 2: A01 * V[9]
    U[1][col] += 1; V[9][col] += 1; W[8][col] += 1
    col += 1
    # Term 3: A10 * V[4]
    U[4][col] += 1; V[4][col] += 1; W[9][col] += 1
    col += 1
    # Term 4: A11 * V[9]
    U[5][col] += 1; V[9][col] += 1; W[9][col] += 1
    col += 1
    
    # A1 * B14 (V[14], V[19])
    # A10=U[2], A11=U[3], A20=U[6], A21=U[7]
    # B24=V[14], B34=V[19]
    
    # Term 5: A20 * V[14]
    U[6][col] += 1; V[14][col] += 1; W[8][col] += 1
    col += 1
    # Term 6: A21 * V[19]
    U[7][col] += 1; V[19][col] += 1; W[8][col] += 1
    col += 1
    # Term 7: A22 * V[14]
    U[6][col] += 1; V[14][col] += 1; W[9][col] += 1
    col += 1
    # Term 8: A23 * V[19]
    U[7][col] += 1; V[19][col] += 1; W[9][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
