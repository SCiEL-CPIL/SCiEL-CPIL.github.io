"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We group the p=10 columns into 5 blocks of size 2.
    # Each block computes a 2x2 product from A (2x3) and B_block (3x2).
    # Rank of 2x3x2 multiplication:
    # We use Strassen for the 2x2x2 part (rank 7) and naive for the remaining 2x1x2 part (rank 4).
    # Total rank per block = 11.
    # Total rank = 5 * 11 = 55.
    
    rank_per_block = 11
    num_blocks = p // 2
    total_rank = rank_per_block * num_blocks
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Coefficients for the 2x3x2 block decomposition
    # A is 2x3: indices 0,1 (row 0), 2,3 (row 1) -> flattened 0..5
    # B is 3x2: indices 0,1 (col 0), 2,3 (col 1) -> flattened 0..5
    # C is 2x2: indices 0,1 (col 0), 2,3 (col 1) -> flattened 0..3
    
    # Strassen terms for A_L (2x2) * B_T (2x2)
    # A_L = [[a0, a1], [a2, a3]] (using 0-based flat indices for 2x2 part)
    # B_T = [[b0, b1], [b2, b3]]
    # Note: In our 2x3 A, the 2x2 part is A[0:2, 0:2] -> flat 0,1,2,3.
    # In our 3x2 B, the 2x2 part is B[0:2, 0:2] -> flat 0,1,2,3.
    
    # Strassen 7 products:
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)b0
    # M3 = a0(b1-b3)
    # M4 = a3(b2-b0)
    # M5 = (a0+a1)b3
    # M6 = (a2-a0)(b0+b1)
    # M7 = (a1-a3)(b2+b3)
    
    # Naive terms for A_R (2x1) * B_B (1x2)
    # A_R = [a4, a5] (col 2 of A)
    # B_B = [b4, b5] (row 2 of B)
    # M8 = a4*b4
    # M9 = a4*b5
    # M10 = a5*b4
    # M11 = a5*b5
    
    # We define the U, V, W coefficients for these 11 terms.
    # U is 6x11, V is 6x11, W is 4x11.
    
    # Term 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = -1; W[2][0] = -1; W[3][0] = 1
    
    # Term 1: M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # Term 2: M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # Term 3: M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # Term 4: M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    # Term 5: M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = -1; W[1][5] = -1
    
    # Term 6: M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[2][6] = -1; W[3][6] = -1
    
    # Term 7: M8 (a4*b4)
    U[4][7] = 1
    V[4][7] = 1
    W[0][7] = 1
    
    # Term 8: M9 (a4*b5)
    U[4][8] = 1
    V[5][8] = 1
    W[1][8] = 1
    
    # Term 9: M10 (a5*b4)
    U[5][9] = 1
    V[4][9] = 1
    W[2][9] = 1
    
    # Term 10: M11 (a5*b5)
    U[5][10] = 1
    V[5][10] = 1
    W[3][10] = 1
    
    # Now, distribute these 11 terms across the 5 blocks.
    # Block b (0 to 4) corresponds to columns 2*b and 2*b+1 of B, and rows 0,1 of C.
    # The A part is the same for all blocks.
    # The B part is shifted by 2*b in the column index.
    # The C part is shifted by 2*b in the column index.
    
    for b in range(num_blocks):
        base_col = 2 * b
        # U is same for all blocks, so we just need to set it once?
        # No, U is shared, but in the tensor decomposition, each term is unique.
        # The terms for block b are distinct from block b+1 because they multiply different B columns.
        # So we need 11 distinct columns in U, V, W for each block.
        
        for k in range(11):
            idx = b * 11 + k
            
            # U part: same for all blocks
            for i in range(6):
                U[i][idx] = U[i][k]
                
            # V part: shifted by base_col in the 3x2 matrix
            # V is 6x... (3x2 flattened)
            # The 2x2 part of B is rows 0,1 cols 0,1.
            # The 1x2 part of B is row 2 cols 0,1.
            # In the full 3x10 B, the block b uses columns 2b, 2b+1.
            # So B_block[0,0] is B[0, 2b], B_block[0,1] is B[0, 2b+1], etc.
            # Flattened index in V: row * 2 + col.
            # V_local[r, c] corresponds to V_global[r * 2 + (2b + c), idx]?
            # No, V is (m*p, rank) = (30, rank).
            # V[row, col] is the coefficient for B[row, col].
            # For block b, the relevant B rows are 0,1,2.
            # The relevant B cols are 2b, 2b+1.
            # The local index (r, c) in 3x2 maps to global (r, 2b+c).
            # So V_global[r * p + (2b+c), idx] = V_local[r, c].
            
            for r in range(3):
                for c in range(2):
                    local_idx = r * 2 + c
                    global_idx = r * p + (2 * b + c)
                    V[global_idx][idx] = V[r * 2 + c][k]
                    
            # W part: shifted by base_col in the 2x2 output
            # W is (p*n, rank) = (20, rank).
            # W[row, col] is the coefficient for C[row, col].
            # For block b, the output is C[0, 2b], C[0, 2b+1], C[1, 2b], C[1, 2b+1].
            # Local index (r, c) in 2x2 maps to global (r, 2b+c).
            # W_global[r * p + (2b+c), idx] = W_local[r, c].
            
            for r in range(2):
                for c in range(2):
                    local_idx = r * 2 + c
                    global_idx = r * p + (2 * b + c)
                    W[global_idx][idx] = W[r * 2 + c][k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
