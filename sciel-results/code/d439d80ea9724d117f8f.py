"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using two layers of Strassen-like logic
    # Layer 1: Split m=4 into two 2s.
    # A (2x4) = [A1, A2] (2x2 each)
    # B (4x8) = [B1; B2] (2x8 each)
    # C = A1 B1 + A2 B2
    # Each term is 2x2 * 2x8.
    # 2x2 * 2x8 can be done with 4 independent 2x2*2x2 Strassen (rank 7 each) -> 28.
    # Total rank 28 + 28 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen forms for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a10)b11
    # M6 = (a11-a00)(b00+b10)
    # M7 = (a01-a11)(b11+b01)
    
    def add_strassen_block(u_offset, v_offset, w_offset, a_idx, b_idx, c_idx):
        # a_idx: list of 4 indices in U (row, col)
        # b_idx: list of 4 indices in V (row, col)
        # c_idx: list of 4 indices in W (row, col)
        # We add 7 forms to the global U, V, W starting at rank offset
        for k in range(7):
            col = u_offset + k
            
            # M1
            if k == 0:
                # (a00+a11) * (b00+b11)
                U[a_idx[0]][col] += 1
                U[a_idx[3]][col] += 1
                V[b_idx[0]][col] += 1
                V[b_idx[3]][col] += 1
                # C00 += M1, C01 += M1, C10 += M1, C11 += M1
                W[c_idx[0]][col] += 1
                W[c_idx[1]][col] += 1
                W[c_idx[2]][col] += 1
                W[c_idx[3]][col] += 1
            # M2
            elif k == 1:
                # (a10+a11) * b00
                U[a_idx[2]][col] += 1
                U[a_idx[3]][col] += 1
                V[b_idx[0]][col] += 1
                # C10 += M2, C11 += M2
                W[c_idx[2]][col] += 1
                W[c_idx[3]][col] += 1
            # M3
            elif k == 2:
                # a00 * (b10-b11)
                U[a_idx[0]][col] += 1
                V[b_idx[2]][col] += 1
                V[b_idx[3]][col] -= 1
                # C10 += M3, C11 -= M3
                W[c_idx[2]][col] += 1
                W[c_idx[3]][col] -= 1
            # M4
            elif k == 3:
                # a11 * (b01-b00)
                U[a_idx[3]][col] += 1
                V[b_idx[1]][col] += 1
                V[b_idx[0]][col] -= 1
                # C01 += M4, C11 -= M4
                W[c_idx[1]][col] += 1
                W[c_idx[3]][col] -= 1
            # M5
            elif k == 4:
                # (a00+a10) * b11
                U[a_idx[0]][col] += 1
                U[a_idx[2]][col] += 1
                V[b_idx[3]][col] += 1
                # C01 += M5, C11 += M5
                W[c_idx[1]][col] += 1
                W[c_idx[3]][col] += 1
            # M6
            elif k == 5:
                # (a11-a00) * (b00+b10)
                U[a_idx[3]][col] += 1
                U[a_idx[0]][col] -= 1
                V[b_idx[0]][col] += 1
                V[b_idx[2]][col] += 1
                # C00 += M6, C10 += M6
                W[c_idx[0]][col] += 1
                W[c_idx[2]][col] += 1
            # M7
            elif k == 6:
                # (a01-a11) * (b11+b01)
                U[a_idx[1]][col] += 1
                U[a_idx[3]][col] -= 1
                V[b_idx[3]][col] += 1
                V[b_idx[1]][col] += 1
                # C00 += M7, C01 += M7
                W[c_idx[0]][col] += 1
                W[c_idx[1]][col] += 1

    # Block 1: A1 (2x2) * B1 (2x8)
    # A1 is rows 0,1 of A. Cols 0,1 of A.
    # In U (8x56), A1 corresponds to U[0:2, 0:2] relative to block?
    # U index: i*m + j.
    # A1: i in 0,1; j in 0,1.
    # U indices: [0*4+0, 0*4+1, 1*4+0, 1*4+1] -> [0, 1, 4, 5]
    a1_idx = [0, 1, 4, 5]
    
    # B1 is rows 0,1 of B. Cols 0,7 of B.
    # In V (32x56), B1 corresponds to V[0:2, 0:8] relative to block?
    # V index: j*p + k.
    # B1: j in 0,1; k in 0,7.
    # We need to split B1 into four 2x2 blocks: B1_1, B1_2, B1_3, B1_4.
    # B1_1: j 0,1; k 0,1. V idx: [0*8+0, 0*8+1, 1*8+0, 1*8+1] -> [0, 1, 8, 9]
    # B1_2: j 0,1; k 2,3. V idx: [2, 3, 10, 11]
    # B1_3: j 0,1; k 4,5. V idx: [4, 5, 12, 13]
    # B1_4: j 0,1; k 6,7. V idx: [6, 7, 14, 15]
    
    b1_blocks = [
        [0, 1, 8, 9],
        [2, 3, 10, 11],
        [4, 5, 12, 13],
        [6, 7, 14, 15]
    ]
    
    # C1 is rows 0,1 of C. Cols 0,7 of C.
    # In W (16x56), C1 corresponds to W[0:2, 0:8] relative to block?
    # W index: k*n + i.
    # C1: k in 0,7; i in 0,1.
    # C1_1: k 0,1; i 0,1. W idx: [0*2+0, 0*2+1, 1*2+0, 1*2+1] -> [0, 1, 2, 3]
    # C1_2: k 2,3; i 0,1. W idx: [4, 5, 6, 7]
    # C1_3: k 4,5; i 0,1. W idx: [8, 9, 10, 11]
    # C1_4: k 6,7; i 0,1. W idx: [12, 13, 14, 15]
    
    c1_blocks = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    # Apply 4 Strassen blocks for A1 * B1
    for b in range(4):
        add_strassen_block(b * 7, b * 7, b * 7, a1_idx, b1_blocks[b], c1_blocks[b])
        
    # Block 2: A2 (2x2) * B2 (2x8)
    # A2: i in 0,1; j in 2,3.
    # U indices: [0*4+2, 0*4+3, 1*4+2, 1*4+3] -> [2, 3, 6, 7]
    a2_idx = [2, 3, 6, 7]
    
    # B2: j in 2,3; k in 0,7.
    # B2_1: j 2,3; k 0,1. V idx: [2*8+0, 2*8+1, 3*8+0, 3*8+1] -> [16, 17, 24, 25]
    # B2_2: j 2,3; k 2,3. V idx: [18, 19, 26, 27]
    # B2_3: j 2,3; k 4,5. V idx: [20, 21, 28, 29]
    # B2_4: j 2,3; k 6,7. V idx: [22, 23, 30, 31]
    
    b2_blocks = [
        [16, 17, 24, 25],
        [18, 19, 26, 27],
        [20, 21, 28, 29],
        [22, 23, 30, 31]
    ]
    
    # C2 is same C1 blocks (since C = C1 + C2)
    # W indices are the same.
    
    # Apply 4 Strassen blocks for A2 * B2
    for b in range(4):
        add_strassen_block(28 + b * 7, 28 + b * 7, 28 + b * 7, a2_idx, b2_blocks[b], c1_blocks[b])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
