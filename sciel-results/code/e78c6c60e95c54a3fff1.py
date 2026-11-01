"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    # Split A (2x3) into A_L (2x2) and A_R (2x1)
    # Split B (3x9) into B_T (2x9) and B_B (1x9)
    # C = A_L B_T + A_R B_B
    # B_T (2x9) split into 4 blocks of 2x2 and 1 block of 2x1
    # A_L B_1 (2x2x2): Rank 7 (Strassen)
    # A_L B_2 (2x2x2): Rank 7
    # A_L B_3 (2x2x2): Rank 7
    # A_L B_4 (2x2x2): Rank 7
    # A_L B_5 (2x2x1): Rank 4
    # A_R B_B (2x1x9): Rank 18
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)] # 6 x 50
    V = [[0] * rank for _ in range(m * p)] # 27 x 50
    W = [[0] * rank for _ in range(p * n)] # 18 x 50
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen decomposition for 2x2x2
    # A indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2) in local 2x2
    # B indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2) in local 2x2
    # C indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2) in local 2x2
    
    def add_strassen(a_base, b_base, c_base):
        # a_base, b_base, c_base are offsets into the flattened arrays
        # U is flattened 2x3. A_L is cols 0,1. 
        # A_L[0,0] -> U[0], A_L[0,1] -> U[1], A_L[1,0] -> U[2], A_L[1,1] -> U[3]
        # Wait, A is 2x3. Flattened row-major:
        # A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2]
        # A[1,0]=U[3], A[1,1]=U[4], A[1,2]=U[5]
        # So A_L uses U[0], U[1], U[3], U[4].
        
        # B is 3x9. Flattened row-major:
        # B[0,0..8] -> V[0..8]
        # B[1,0..8] -> V[9..17]
        # B[2,0..8] -> V[18..26]
        # B_T uses rows 0,1.
        # B_1 (cols 0,1): V[0], V[1], V[9], V[10]
        
        # C is 2x9. Flattened row-major:
        # C[0,0..8] -> W[0..8]
        # C[1,0..8] -> W[9..17]
        # C_1 (cols 0,1): W[0], W[1], W[9], W[10]

        # Strassen terms
        # P1 = (A00+A11)(B00+B11)
        add_term(a_base+0, b_base+0, c_base+0, 1, 1, 1)
        add_term(a_base+1, b_base+1, c_base+0, 1, 1, 1) # Wait, A11 is index 1? 
        # Let's map local indices carefully.
        # Local A: 00, 01, 10, 11
        # Global U for A_L:
        # 00 -> 0
        # 01 -> 1
        # 10 -> 3
        # 11 -> 4
        
        # Local B: 00, 01, 10, 11
        # Global V for B_block:
        # 00 -> b_base + 0
        # 01 -> b_base + 1
        # 10 -> b_base + 9 (next row)
        # 11 -> b_base + 10
        
        # Local C: 00, 01, 10, 11
        # Global W for C_block:
        # 00 -> c_base + 0
        # 01 -> c_base + 1
        # 10 -> c_base + 9
        # 11 -> c_base + 10

        # Reset col for this block? No, col is global.
        # I need to define the terms explicitly.
        
        # P1
        # (A00 + A11) (B00 + B11)
        # U: A00(1), A11(1)
        # V: B00(1), B11(1)
        # W: C00(1), C01(1), C10(1), C11(1) -- No, W is linear combination.
        # The decomposition is C = sum u_r v_r w_r.
        # For Strassen, we have 7 products.
        # Each product contributes to C.
        # I need to add 7 columns to U, V, W.
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C01=1, C10=1, C11=1? 
        # No, the result of this product is added to C00, C01, C10, C11?
        # Strassen:
        # C00 = P1 - P4 + P6
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P3 + P7
        
        # So for P1:
        # W: C00=1, C11=1
        # Wait, P1 contributes to C00 and C11?
        # Let's check:
        # C00 = P1 - P4 + P6
        # C11 = P1 - P3 + P7
        # So P1 has +1 in W[C00] and +1 in W[C11].
        
        # Let's implement the 7 terms for one block.
        
        # Block 1 (cols 0-1)
        # b_base = 0, c_base = 0
        # U indices for A_L: 0, 1, 3, 4
        # V indices for B_1: 0, 1, 9, 10
        # W indices for C_1: 0, 1, 9, 10
        
        # P1
        U[0][col]=1; U[4][col]=1; V[0][col]=1; V[10][col]=1; W[0][col]=1; W[10][col]=1; col+=1
        # P2: (A10+A11) B00
        # C10 += P2
        U[3][col]=1; U[4][col]=1; V[0][col]=1; W[9][col]=1; col+=1
        # P3: A00 (B10-B11)
        # C01 += P3, C11 -= P3
        U[0][col]=1; V[9][col]=1; V[10][col]=-1; W[1][col]=1; W[10][col]=-1; col+=1
        # P4: A11 (B00-B01)
        # C00 -= P4, C10 += P4
        U[4][col]=1; V[0][col]=1; V[1][col]=-1; W[0][col]=-1; W[9][col]=1; col+=1
        # P5: (A00+A01) B11
        # C01 += P5
        U[0][col]=1; U[1][col]=1; V[10][col]=1; W[1][col]=1; col+=1
        # P6: (A10-A00) (B01+B00)
        # C00 += P6
        U[3][col]=1; U[0][col]=-1; V[1][col]=1; V[0][col]=1; W[0][col]=1; col+=1
        # P7: (A01-A11) (B10+B11)
        # C11 += P7
        U[1][col]=1; U[4][col]=-1; V[9][col]=1; V[10][col]=1; W[10][col]=1; col+=1
        
        # Block 2 (cols 2-3)
        # b_base = 2, c_base = 2
        # V indices: 2, 3, 11, 12
        # W indices: 2, 3, 11, 12
        
        U[0][col]=1; U[4][col]=1; V[2][col]=1; V[12][col]=1; W[2][col]=1; W[12][col]=1; col+=1
        U[3][col]=1; U[4][col]=1; V[2][col]=1; W[11][col]=1; col+=1
        U[0][col]=1; V[11][col]=1; V[12][col]=-1; W[3][col]=1; W[12][col]=-1; col+=1
        U[4][col]=1; V[2][col]=1; V[3][col]=-1; W[2][col]=-1; W[11][col]=1; col+=1
        U[0][col]=1; U[1][col]=1; V[12][col]=1; W[3][col]=1; col+=1
        U[3][col]=1; U[0][col]=-1; V[3][col]=1; V[2][col]=1; W[2][col]=1; col+=1
        U[1][col]=1; U[4][col]=-1; V[11][col]=1; V[12][col]=1; W[12][col]=1; col+=1

        # Block 3 (cols 4-5)
        # b_base = 4, c_base = 4
        # V indices: 4, 5, 13, 14
        # W indices: 4, 5, 13, 14
        
        U[0][col]=1; U[4][col]=1; V[4][col]=1; V[14][col]=1; W[4][col]=1; W[14][col]=1; col+=1
        U[3][col]=1; U[4][col]=1; V[4][col]=1; W[13][col]=1; col+=1
        U[0][col]=1; V[13][col]=1; V[14][col]=-1; W[5][col]=1; W[14][col]=-1; col+=1
        U[4][col]=1; V[4][col]=1; V[5][col]=-1; W[4][col]=-1; W[13][col]=1; col+=1
        U[0][col]=1; U[1][col]=1; V[14][col]=1; W[5][col]=1; col+=1
        U[3][col]=1; U[0][col]=-1; V[5][col]=1; V[4][col]=1; W[4][col]=1; col+=1
        U[1][col]=1; U[4][col]=-1; V[13][col]=1; V[14][col]=1; W[14][col]=1; col+=1

        # Block 4 (cols 6-7)
        # b_base = 6, c_base = 6
        # V indices: 6, 7, 15, 16
        # W indices: 6, 7, 15, 16
        
        U[0][col]=1; U[4][col]=1; V[6][col]=1; V[16][col]=1; W[6][col]=1; W[16][col]=1; col+=1
        U[3][col]=1; U[4][col]=1; V[6][col]=1; W[15][col]=1; col+=1
        U[0][col]=1; V[15][col]=1; V[16][col]=-1; W[7][col]=1; W[16][col]=-1; col+=1
        U[4][col]=1; V[6][col]=1; V[7][col]=-1; W[6][col]=-1; W[15][col]=1; col+=1
        U[0][col]=1; U[1][col]=1; V[16][col]=1; W[7][col]=1; col+=1
        U[3][col]=1; U[0][col]=-1; V[7][col]=1; V[6][col]=1; W[6][col]=1; col+=1
        U[1][col]=1; U[4][col]=-1; V[15][col]=1; V[16][col]=1; W[16][col]=1; col+=1

        # Block 5 (col 8) - 2x2x1
        # A_L (2x2) * B_5 (2x1)
        # B_5 uses V[8] (row 0, col 8) and V[17] (row 1, col 8)
        # C_5 uses W[8] (row 0, col 8) and W[17] (row 1, col 8)
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        # Term 1: A00 B00 -> C00
        U[0][col]=1; V[8][col]=1; W[8][col]=1; col+=1
        # Term 2: A01 B10 -> C00
        U[1][col]=1; V[17][col]=1; W[8][col]=1; col+=1
        # Term 3: A10 B00 -> C10
        U[3][col]=1; V[8][col]=1; W[17][col]=1; col+=1
        # Term 4: A11 B10 -> C10
        U[4][col]=1; V[17][col]=1; W[17][col]=1; col+=1

        # A_R (2x1) * B_B (1x9)
        # A_R uses U[2] (row 0, col 2) and U[5] (row 1, col 2)
        # B_B uses V[18..26] (row 2, cols 0..8)
        # C uses W[0..8] (row 0) and W[9..17] (row 1)
        
        # C0k = A02 B2k
        # C1k = A12 B2k
        
        for k in range(9):
            # Term for C0k
            U[2][col]=1; V[18+k][col]=1; W[k][col]=1; col+=1
            # Term for C1k
            U[5][col]=1; V[18+k][col]=1; W[9+k][col]=1; col+=1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
