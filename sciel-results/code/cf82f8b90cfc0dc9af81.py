"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 (Strassen composed) + Rank 16 for 4x4x1 (Standard)
    rank = 49 + 16
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 multiplication (Rank 49) ---
    # Strassen 2x2x2 base case (Rank 7)
    # Inputs: A (2x2), B (2x2)
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # U_2, V_2, W_2 for Strassen 2x2x2
    # col 0: m1 = (a00+a11)(b00+b11)
    # col 1: m2 = (a10+a11)b00
    # col 2: m3 = a00(b01-b11)
    # col 3: m4 = a11(b10-b00)
    # col 4: m5 = (a00+a01)b11
    # col 5: m6 = (a10-a00)(b00+b01)
    # col 6: m7 = (a01-a11)(b10+b11)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # m1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[1][0] = 1; W2[2][0] = 1; W2[3][0] = 1 # c00, c01, c10, c11 all use m1? 
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    W2[0][0] = 1  # c00
    W2[1][0] = 0  # c01
    W2[2][0] = 0  # c10
    W2[3][0] = 1  # c11
    
    # m2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[0][1] = 0
    W2[1][1] = 0
    W2[2][1] = 1  # c10
    W2[3][1] = -1 # c11
    
    # m3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[0][2] = 0
    W2[1][2] = 1  # c01
    W2[2][2] = 0
    W2[3][2] = 1  # c11
    
    # m4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1  # c00
    W2[1][3] = 0
    W2[2][3] = 1  # c10
    W2[3][3] = 0
    
    # m5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1 # c00
    W2[1][4] = 1  # c01
    W2[2][4] = 0
    W2[3][4] = 0
    
    # m6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[0][5] = 0
    W2[1][5] = 0
    W2[2][5] = 0
    W2[3][5] = 1  # c11
    
    # m7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1  # c00
    W2[1][6] = 0
    W2[2][6] = 0
    W2[3][6] = 0
    
    # Compose 4x4x4 using Strassen on blocks
    # Block indices: 00, 01, 10, 11
    # A_block_00: rows 0-1, cols 0-1 -> U indices 0-3
    # A_block_01: rows 0-1, cols 2-3 -> U indices 4-7
    # A_block_10: rows 2-3, cols 0-1 -> U indices 8-11
    # A_block_11: rows 2-3, cols 2-3 -> U indices 12-15
    
    # We iterate over the 7 Strassen "multiplications" at the block level
    # For each block multiplication, we add the 7 scalar multiplications from U2, V2, W2
    
    block_U_offsets = [0, 4, 8, 12] # 00, 01, 10, 11
    block_V_offsets = [0, 4, 8, 12] # Note: B is 4x4, so m=4, p=4 for this part.
                                    # B_block_00: rows 0-1, cols 0-1 -> V indices 0-3 (j*p+k, j=0-1, k=0-1)
                                    # B_block_01: rows 0-1, cols 2-3 -> V indices 4-7
                                    # B_block_10: rows 2-3, cols 0-1 -> V indices 8-11
                                    # B_block_11: rows 2-3, cols 2-3 -> V indices 12-15
    block_W_offsets = [0, 4, 8, 12] # C is 4x4.
                                    # C_block_00: rows 0-1, cols 0-1 -> W indices 0-3 (k*n+i, k=0-1, i=0-1)
                                    # C_block_01: rows 0-1, cols 2-3 -> W indices 4-7
                                    # C_block_10: rows 2-3, cols 0-1 -> W indices 8-11
                                    # C_block_11: rows 2-3, cols 2-3 -> W indices 12-15
    
    # Strassen Block Formulas:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00      -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)      -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)      -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11      -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01)-> C11 += M6
    # M7 = (A01-A11)(B10+B11)-> C00 += M7
    
    # We need to map these to the 49 scalar multiplications.
    # Each block multiplication M_k is itself a 2x2x2 multiplication with rank 7.
    # So we have 7 block mults * 7 scalar mults = 49.
    
    # Let's define the coefficients for the block-level Strassen steps
    # block_mults: list of (U_blocks, V_blocks, W_blocks, coeff)
    # U_blocks: list of (block_idx, coeff)
    # V_blocks: list of (block_idx, coeff)
    # W_blocks: list of (block_idx, coeff)
    
    # M1: (A00+A11)(B00+B11)
    # U: 00:1, 11:1
    # V: 00:1, 11:1
    # W: 00:1, 11:1
    # M2: (A10+A11)B00
    # U: 10:1, 11:1
    # V: 00:1
    # W: 10:1, 11:-1
    # M3: A00(B01-B11)
    # U: 00:1
    # V: 01:1, 11:-1
    # W: 01:1, 11:1
    # M4: A11(B10-B00)
    # U: 11:1
    # V: 10:1, 00:-1
    # W: 00:1, 10:1
    # M5: (A00+A01)B11
    # U: 00:1, 01:1
    # V: 11:1
    # W: 00:-1, 01:1
    # M6: (A10-A00)(B00+B01)
    # U: 10:1, 00:-1
    # V: 00:1, 01:1
    # W: 11:1
    # M7: (A01-A11)(B10+B11)
    # U: 01:1, 11:-1
    # V: 10:1, 11:1
    # W: 00:1
    
    block_defs = [
        ([ (0,1), (3,1) ], [ (0,1), (3,1) ], [ (0,1), (3,1) ]), # M1
        ([ (2,1), (3,1) ], [ (0,1) ], [ (2,1), (3,-1) ]),       # M2
        ([ (0,1) ], [ (1,1), (3,-1) ], [ (1,1), (3,1) ]),       # M3
        ([ (3,1) ], [ (2,1), (0,-1) ], [ (0,1), (2,1) ]),       # M4
        ([ (0,1), (1,1) ], [ (3,1) ], [ (0,-1), (1,1) ]),       # M5
        ([ (2,1), (0,-1) ], [ (0,1), (1,1) ], [ (3,1) ]),       # M6
        ([ (1,1), (3,-1) ], [ (2,1), (3,1) ], [ (0,1) ])        # M7
    ]
    
    col_offset = 0
    for b_u, b_v, b_w in block_defs:
        # For each scalar multiplication in the 2x2x2 base case (7 of them)
        for s in range(7):
            # U contribution
            for u_idx, u_coeff in b_u:
                # u_idx is block index (0-3)
                # u_coeff is coefficient
                # The scalar multiplication s has U2[i][s] for i in 0-3
                for i in range(4):
                    if U2[i][s] != 0:
                        # Global U index: block_offset + i
                        global_u_idx = block_U_offsets[u_idx] + i
                        U[global_u_idx][col_offset] += u_coeff * U2[i][s]
                        
            # V contribution
            for v_idx, v_coeff in b_v:
                for j in range(4):
                    if V2[j][s] != 0:
                        global_v_idx = block_V_offsets[v_idx] + j
                        V[global_v_idx][col_offset] += v_coeff * V2[j][s]
                        
            # W contribution
            for w_idx, w_coeff in b_w:
                for k in range(4):
                    if W2[k][s] != 0:
                        global_w_idx = block_W_offsets[w_idx] + k
                        W[global_w_idx][col_offset] += w_coeff * W2[k][s]
                        
            col_offset += 1
            
    # --- Part 2: 4x4x1 multiplication (Rank 16) ---
    # C[:, 4] = A * B[:, 4]
    # k=4 is the 5th column (index 4)
    # W index: k*n + i = 4*4 + i = 16 + i
    # V index: j*p + k = j*5 + 4
    # U index: i*m + j = i*4 + j
    
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col_offset] = 1
            V[j*5 + 4][col_offset] = 1
            W[16 + i][col_offset] = 1
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
