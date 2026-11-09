"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    # Split A into A1 (2x2) and A2 (2x1)
    # Split B into B1 (2x6) and B2 (1x6)
    # C = A1*B1 + A2*B2
    
    # A1*B1: 2x2 * 2x6. Split B1 into three 2x2 blocks.
    # Each 2x2*2x2 uses Strassen (rank 7). Total 21.
    # A2*B2: 2x1 * 1x6. 12 products. Total 12.
    # Total rank 33.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)] # 6 x 33
    V = [[0] * rank for _ in range(m * p)] # 18 x 33
    W = [[0] * rank for _ in range(p * n)] # 12 x 33
    
    col = 0
    
    # Part 1: A1 * B1 (2x2 * 2x6)
    # A1 indices: A[0,0], A[0,1], A[1,0], A[1,1] -> 0, 1, 2, 3
    # B1 indices: B[0,0..5], B[1,0..5] -> 0..11
    # C indices: C[0,0..5], C[1,0..5] -> 0..11
    
    # Strassen for 2x2*2x2
    # m0 = a00 * b00
    # m1 = a10 * (b01 + b11) ... etc
    # Standard Strassen:
    # M1 = a00 * b00
    # M2 = a10 * b01
    # M3 = a01 * b10
    # M4 = a11 * b11
    # M5 = (a00+a10) * b01
    # M6 = (a01+a11) * (b00+b10)
    # M7 = (a00+a11) * (b01+b11)
    # c00 = M1 - M5 + M6 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M4 + M7
    
    # We have 3 blocks for B1.
    for block in range(3):
        # B block columns: block*2, block*2+1
        # B row 0: indices 0, 1, 2, 3, 4, 5
        # B row 1: indices 6, 7, 8, 9, 10, 11
        
        # M1
        U[0][col] = 1 # a00
        V[0][col] = 1 # b00 (B[0, block*2])
        W[block*2][col] = 1 # c00 (C[0, block*2])
        W[6+block*2][col] = 1 # c11 (C[1, block*2+1])
        col += 1
        
        # M2
        U[2][col] = 1 # a10
        V[1][col] = 1 # b01 (B[0, block*2+1])
        W[6+block*2][col] = -1 # c10 (C[1, block*2])
        col += 1
        
        # M3
        U[1][col] = 1 # a01
        V[6][col] = 1 # b10 (B[1, block*2])
        W[block*2+1][col] = 1 # c01 (C[0, block*2+1])
        col += 1
        
        # M4
        U[3][col] = 1 # a11
        V[7][col] = 1 # b11 (B[1, block*2+1])
        W[6+block*2][col] = 1 # c10 (C[1, block*2])
        W[6+block*2+1][col] = 1 # c11 (C[1, block*2+1])
        col += 1
        
        # M5
        U[0][col] = 1 # a00
        U[2][col] = 1 # a10
        V[1][col] = 1 # b01
        W[block*2+1][col] = 1 # c01
        W[block*2][col] = -1 # c00
        col += 1
        
        # M6
        U[1][col] = 1 # a01
        U[3][col] = 1 # a11
        V[0][col] = 1 # b00
        V[6][col] = 1 # b10
        W[block*2][col] = 1 # c00
        col += 1
        
        # M7
        U[0][col] = 1 # a00
        U[3][col] = 1 # a11
        V[1][col] = 1 # b01
        V[7][col] = 1 # b11
        W[block*2][col] = 1 # c00
        W[6+block*2+1][col] = 1 # c11
        col += 1

    # Part 2: A2 * B2 (2x1 * 1x6)
    # A2 indices: A[0,2], A[1,2] -> 4, 5
    # B2 indices: B[2,0..5] -> 12..17
    # C indices: C[0,0..5], C[1,0..5] -> 0..11
    
    for j in range(6):
        # C[0, j] += A[0,2] * B[2, j]
        U[4][col] = 1
        V[12+j][col] = 1
        W[j][col] = 1
        col += 1
        
        # C[1, j] += A[1,2] * B[2, j]
        U[5][col] = 1
        V[12+j][col] = 1
        W[6+j][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
