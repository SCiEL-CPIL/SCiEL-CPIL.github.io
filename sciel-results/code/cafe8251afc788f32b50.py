import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # Split m=4 into m1=2, m2=2.
    # A is (3, 4) -> A1 (3, 2), A2 (3, 2)
    # B is (4, 6) -> B1 (2, 6), B2 (2, 6)
    # AB = A1 B1 + A2 B2
    
    # For each product A_i (3, 2) B_i (2, 6):
    # Split A_i (3, 2) into A_top (2, 2) and A_bot (1, 2)
    # A_i B_i = A_top B_i + A_bot B_i
    
    # A_top (2, 2) B_i (2, 6):
    # Split B_i (2, 6) into 3 blocks of (2, 2): B_i1, B_i2, B_i3
    # A_top B_i = sum_{j=1..3} A_top B_ij
    # Each is 2x2 by 2x2, rank 7 (Strassen)
    # Total rank for this part: 3 * 7 = 21
    
    # A_bot (1, 2) B_i (2, 6):
    # Rank 1*2*6 = 12 (Schoolbook)
    
    # Total rank for one A_i B_i: 21 + 12 = 33
    # Total rank for AB: 33 + 33 = 66
    
    rank = 66
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen multipliers for 2x2 matrix multiplication
    # Inputs X (2x2), Y (2x2). Output Z (2x2).
    # U rows correspond to X elements, V rows to Y elements, W rows to Z elements.
    # Z = XY
    # x00, x01, x10, x11
    # y00, y01, y10, y11
    # z00, z01, z10, z11
    
    def add_strassen_2x2(offset, x_base, y_base, z_base):
        # x_base: index in U for first element of 2x2 X
        # y_base: index in V for first element of 2x2 Y
        # z_base: index in W for first element of 2x2 Z
        # X indices: x_base, x_base+1, x_base+2, x_base+3 (row-major)
        # Y indices: y_base, y_base+1, y_base+2, y_base+3
        # Z indices: z_base, z_base+1, z_base+2, z_base+3
        
        # Strassen's 7 multipliers
        # m1 = (x00 + x11)(y00 + y11)
        # m2 = (x10 + x11)y00
        # m3 = x00(y01 - y11)
        # m4 = x11(y10 - y00)
        # m5 = (x00 + x01)y11
        # m6 = (x10 - x00)(y00 + y01)
        # m7 = (x01 - x11)(y10 + y11)
        
        # Output
        # z00 = m1 + m4 - m5 + m7
        # z01 = m3 + m5
        # z10 = m2 + m4
        # z11 = m1 - m2 + m3 + m6
        
        # We need to set U, V, W such that sum(U[:,c] V[:,c]) = W[:,c] contribution?
        # No, the tensor decomposition is U[i,c] V[j,c] W[k,c] = 1 if term c contributes to Z_k.
        # Actually, the rank-1 term is u v w^T.
        # Here we are constructing the full U, V, W matrices.
        
        # Each Strassen multiplier m_k is a linear combination of X and Y.
        # This means it's not a single rank-1 term in the original variables,
        # but a product of two linear forms.
        # The decomposition of the tensor for 2x2x2 into 7 rank-1 terms is:
        # T = sum_{k=1..7} u_k v_k w_k^T
        
        # The 7 rank-1 terms for 2x2x2:
        # 1. (x00+x11)(y00+y11) -> z00, z11
        # 2. (x10+x11)y00 -> z10
        # 3. x00(y01-y11) -> z01
        # 4. x11(y10-y00) -> z10, z00? No.
        
        # Let's use the explicit rank-1 vectors.
        # Term 1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) ? No.
        
        # Correct Strassen Rank-1 Decomposition:
        # m1 = (x00+x11)(y00+y11)
        # m2 = (x10+x11)y00
        # m3 = x00(y01-y11)
        # m4 = x11(y10-y00)
        # m5 = (x00+x01)y11
        # m6 = (x10-x00)(y00+y01)
        # m7 = (x01-x11)(y10+y11)
        
        # z00 = m1 + m4 - m5 + m7
        # z01 = m3 + m5
        # z10 = m2 + m4
        # z11 = m1 - m2 + m3 + m6
        
        # We add 7 columns to U, V, W.
        
        # Col 0: m1
        U[x_base][offset] = 1; U[x_base+3][offset] = 1
        V[y_base][offset] = 1; V[y_base+3][offset] = 1
        W[z_base][offset] = 1; W[z_base+3][offset] = 1
        
        # Col 1: m2
        U[x_base+2][offset+1] = 1; U[x_base+3][offset+1] = 1
        V[y_base][offset+1] = 1
        W[z_base+2][offset+1] = 1
        
        # Col 2: m3
        U[x_base][offset+2] = 1
        V[y_base+1][offset+2] = 1; V[y_base+3][offset+2] = -1
        W[z_base+1][offset+2] = 1
        
        # Col 3: m4
        U[x_base+3][offset+3] = 1
        V[y_base+2][offset+3] = 1; V[y_base][offset+3] = -1
        W[z_base][offset+3] = 1; W[z_base+2][offset+3] = 1
        
        # Col 4: m5
        U[x_base][offset+4] = 1; U[x_base+1][offset+4] = 1
        V[y_base+3][offset+4] = 1
        W[z_base][offset+4] = -1; W[z_base+1][offset+4] = 1
        
        # Col 5: m6
        U[x_base+2][offset+5] = 1; U[x_base][offset+5] = -1
        V[y_base][offset+5] = 1; V[y_base+1][offset+5] = 1
        W[z_base+3][offset+5] = 1
        
        # Col 6: m7
        U[x_base+1][offset+6] = 1; U[x_base+3][offset+6] = -1
        V[y_base+2][offset+6] = 1; V[y_base+3][offset+6] = 1
        W[z_base][offset+6] = 1; W[z_base+3][offset+6] = 1

    def add_schoolbook_1x2x6(offset, a_base, b_base, c_base):
        # a: 1x2, b: 2x6, c: 1x6
        # Rank 12
        # a0, a1
        # b00..b05, b10..b15
        # c0..c5
        # c_j = a0 b0j + a1 b1j
        # 12 terms:
        for j in range(6):
            # Term a0 b0j -> c_j
            U[a_base][offset + 2*j] = 1
            V[b_base + j][offset + 2*j] = 1
            W[c_base + j][offset + 2*j] = 1
            
            # Term a1 b1j -> c_j
            U[a_base+1][offset + 2*j + 1] = 1
            V[b_base + 2 + j][offset + 2*j + 1] = 1
            W[c_base + j][offset + 2*j + 1] = 1

    # Process A1 B1
    # A1 is rows 0,1,2 of A (3x2). In U, rows 0..5 (3*2).
    # B1 is cols 0..1 of B (2x6). In V, rows 0..11 (2*6).
    # Result is part of Z.
    
    # A1_top (2x2) is A rows 0,1. U indices 0..3.
    # A1_bot (1x2) is A row 2. U index 4,5.
    
    # B1 (2x6). V indices 0..11.
    # B1 is split into 3 2x2 blocks.
    # Block 0: V 0..3
    # Block 1: V 4..7
    # Block 2: V 8..11
    
    # Z1 (3x2) is part of output.
    # Output Z is 6x3? No, W is p*n = 6*3 = 18.
    # Indexing W: W[k*n+i, r]. k is col of Z, i is row of Z.
    # Z1 corresponds to cols 0,1 of Z. Rows 0,1,2.
    # W indices for Z1:
    # Row 0, Col 0: 0*3 + 0 = 0
    # Row 0, Col 1: 1*3 + 0 = 3
    # Row 1, Col 0: 0*3 + 1 = 1
    # Row 1, Col 1: 1*3 + 1 = 4
    # Row 2, Col 0: 0*3 + 2 = 2
    # Row 2, Col 1: 1*3 + 2 = 5
    
    # A1_top B1: 2x2 by 2x6.
    # Top 2 rows of Z (Rows 0,1). Cols 0,1.
    # W indices for 2x2 block (Rows 0,1 Cols 0,1): 0, 3, 1, 4.
    
    # Strassen 1: A1_top * B1_block0 (V 0..3) -> Z_block0 (W 0,3,1,4)
    add_strassen_2x2(0, 0, 0, 0) # x_base=0, y_base=0, z_base=0
    # Strassen 2: A1_top * B1_block1 (V 4..7) -> Z_block1 (W 3,6,4,7)?
    # Z_col 1. Rows 0,1.
    # W indices:
    # Row 0 Col 1: 1*3+0 = 3
    # Row 1 Col 1: 1*3+1 = 4
    # Row 2? No, A_top is only 2 rows.
    # So Z_block1 is 2x1? No, B_block is 2x2.
    # B1 is 2x6. Split into 2x2, 2x2, 2x2.
    # Block 1 (cols 2,3 of B) -> Z cols 2,3.
    # Wait, A is 3x2. B is 2x6.
    # A_top is 2x2. B1_block1 is 2x2.
    # Result is 2x2.
    # Corresponds to Z Rows 0,1 and Cols 2,3.
    # W indices:
    # Row 0 Col 2: 2*3+0 = 6
    # Row 0 Col 3: 3*3+0 = 9
    # Row 1 Col 2: 2*3+1 = 7
    # Row 1 Col 3: 3*3+1 = 10
    
    add_strassen_2x2(7, 0, 4, 6) # x_base=0 (A_top same), y_base=4, z_base=6 (Row 0 Col 2)
    
    # Block 2 (cols 4,5 of B) -> Z cols 4,5.
    # W indices:
    # Row 0 Col 4: 4*3+0 = 12
    # Row 0 Col 5: 5*3+0 = 15
    # Row 1 Col 4: 4*3+1 = 13
    # Row 1 Col 5: 5*3+1 = 16
    
    add_strassen_2x2(14, 0, 8, 12)
    
    # A1_bot B1: 1x2 by 2x6.
    # A1_bot is Row 2 of A. U indices 4,5.
    # B1 is V 0..11.
    # Result is Row 2 of Z, Cols 0..5.
    # W indices:
    # Row 2 Col 0: 0*3+2 = 2
    # Row 2 Col 1: 1*3+2 = 5
    # Row 2 Col 2: 2*3+2 = 8
    # Row 2 Col 3: 3*3+2 = 11
    # Row 2 Col 4: 4*3+2 = 14
    # Row 2 Col 5: 5*3+2 = 17
    
    add_schoolbook_1x2x6(21, 4, 0, 2)
    
    # Process A2 B2
    # A2 is rows 0,1,2 of A but corresponding to B2.
    # Wait, A is [A1, A2].
    # A1 uses A cols 0,1. A2 uses A cols 2,3.
    # U indices for A2:
    # Row 0 Col 2: 0*4+2 = 2? No, U is n*m = 12.
    # A is 3x4.
    # A1: Cols 0,1. Indices 0,1 (Row 0), 4,5 (Row 1), 8,9 (Row 2).
    # A2: Cols 2,3. Indices 2,3 (Row 0), 6,7 (Row 1), 10,11 (Row 2).
    
    # B2 is rows 2,3 of B. V indices 12..23.
    
    # A2_top (2x2): Rows 0,1 of A, Cols 2,3. U indices 2,3 and 6,7.
    # A2_bot (1x2): Row 2 of A, Cols 2,3. U indices 10,11.
    
    # B2 (2x6): V indices 12..23.
    # Blocks: 12..15, 16..19, 20..23.
    
    # Z2: Cols 0..5 of Z?
    # No, AB = A1 B1 + A2 B2.
    # B1 and B2 are both 2x6.
    # The result of A1 B1 is a 3x6 matrix.
    # The result of A2 B2 is a 3x6 matrix.
    # They are added element-wise.
    # So they contribute to the same W indices.
    
    # A2_top B2_block0:
    # x_base = 2 (A2_top starts at U index 2)
    # y_base = 12
    # z_base = 0 (Same Z positions)
    
    add_strassen_2x2(28, 2, 12, 0)
    
    # A2_top B2_block1:
    # z_base = 6
    
    add_strassen_2x2(35, 2, 16, 6)
    
    # A2_top B2_block2:
    # z_base = 12
    
    add_strassen_2x2(42, 2, 20, 12)
    
    # A2_bot B2:
    # x_base = 10
    # y_base = 12
    # z_base = 2
    
    add_schoolbook_1x2x6(49, 10, 12, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
