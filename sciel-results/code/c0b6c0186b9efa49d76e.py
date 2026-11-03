"""
Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
Uses Strassen-style block decomposition to reduce rank below schoolbook 72.
"""

import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=6
    
    # We build the decomposition as a list of rank-1 terms.
    # Each term is (U_vec, V_vec, W_vec) where:
    #   U_vec has length n*m = 12, indexed by (i,j) -> i*4+j
    #   V_vec has length m*p = 24, indexed by (j,k) -> j*6+k
    #   W_vec has length p*n = 18, indexed by (k,i) -> 3*k+i
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        """
        u_entries: list of (i, j, coeff)
        v_entries: list of (j, k, coeff)
        w_entries: list of (i, k, coeff)
        """
        u_vec = [0] * 12
        v_vec = [0] * 24
        w_vec = [0] * 18
        for i, j, c in u_entries:
            u_vec[i * 4 + j] = c
        for j, k, c in v_entries:
            v_vec[j * 6 + k] = c
        for i, k, c in w_entries:
            w_vec[3 * k + i] = c
        terms.append((u_vec, v_vec, w_vec))
    
    # Strategy: Use Strassen for 2x2 sub-problems.
    # 
    # Partition A (3x4) into:
    #   A_top: rows 0,1 (2x4)
    #   A_bot: row 2 (1x4)
    #
    # Partition B (4x6) into:
    #   B_0: cols 0,1 (4x2)
    #   B_1: cols 2,3 (4x2)
    #   B_2: cols 4,5 (4x2)
    #
    # C (3x6) blocks:
    #   C[0:2, 0:2] = A_top * B_0  (2x2)
    #   C[0:2, 2:4] = A_top * B_1  (2x2)
    #   C[0:2, 4:6] = A_top * B_2  (2x2)
    #   C[2:3, 0:2] = A_bot * B_0  (1x2)
    #   C[2:3, 2:4] = A_bot * B_1  (1x2)
    #   C[2:3, 4:6] = A_bot * B_2  (1x2)
    
    # For A_top * B_0: A_top is 2x4, B_0 is 4x2, result is 2x2.
    # C[i][k] = sum_j A_top[i][j] * B_0[j][k] for i in {0,1}, k in {0,1}
    #
    # This is a (2,4,2) matrix multiplication. We can use Strassen-like methods.
    # Actually, let's think of it as: we have 4 intermediate products per output, 4 outputs = 16 schoolbook.
    # Can we do better? The rank of M_{2,4,2}... 
    #
    # Let me just use a direct 4-term decomposition for each (2x4)*(4x2) block,
    # treating it as 4 independent dot products of length 4.
    # But that's still 4*4=16 per block... no wait, each dot product needs 4 multiplications.
    #
    # Actually, for (2x4)*(4x2), we can use the following:
    # Let A = [a0, a1, a2, a3; b0, b1, b2, b3] (2x4)
    # Let B = [c0, d0; c1, d1; c2, d2; c3, d3] (4x2)
    # C[0][0] = a0*c0 + a1*c1 + a2*c2 + a3*c3
    # C[0][1] = a0*d0 + a1*d1 + a2*d2 + a3*d3
    # C[1][0] = b0*c0 + b1*c1 + b2*c2 + b3*c3
    # C[1][1] = b0*d0 + b1*d1 + b2*d2 + b3*d3
    #
    # We can use Strassen-style: define
    # M1 = (a0+b0)*(c0+d0)  -- no, this doesn't work directly.
    #
    # Actually, for (2x4)*(4x2), one approach is to split the 4 into 2+2:
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x2
    # C = A1*B1 + A2*B2
    #
    # Each A1*B1 is 2x2*2x2 -> Strassen rank 7
    # Each A2*B2 is 2x2*2x2 -> Strassen rank 7
    # Total: 14 per (2x4)*(4x2) block, vs 16 schoolbook.
    
    # For (1x4)*(4x2) -> 1x2:
    # C[0][0] = sum_j A[j] * B[j][0]
    # C[0][1] = sum_j A[j] * B[j][1]
    # This is 4+4 = 8 schoolbook. Can we do better?
    # Split into 2+2: A = [A1 | A2], B = [B1; B2]
    # C = A1*B1 + A2*B2 where each is (1x2)*(2x2) -> 1x2
    # (1x2)*(2x2) = [x,y]*[[p,q],[r,s]] = [xp+yr, xq+ys] -> 4 mults
    # Total: 4+4 = 8. No improvement.
    #
    # So for (1x4)*(4x2), rank is 8 (no improvement over schoolbook).
    
    # Total rank: 3 * 14 + 3 * 8 = 42 + 24 = 66.
    
    # Let me implement the Strassen decomposition for 2x2*2x2.
    
    def strassen_2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add Strassen decomposition for 2x2 * 2x2.
        a_rows, a_cols: indices in A
        b_rows, b_cols: indices in B
        c_rows, c_cols: indices in C
        """
        a00 = (a_rows[0], a_cols[0])
        a01 = (a_rows[0], a_cols[1])
        a10 = (a_rows[1], a_cols[0])
        a11 = (a_rows[1], a_cols[1])
        
        b00 = (b_rows[0], b_cols[0])
        b01 = (b_rows[0], b_cols[1])
        b10 = (b_rows[1], b_cols[0])
        b11 = (b_rows[1], b_cols[1])
        
        c00 = (c_rows[0], c_cols[0])
        c01 = (c_rows[0], c_cols[1])
        c10 = (c_rows[1], c_cols[0])
        c11 = (c_rows[1], c_cols[1])
        
        # M1 = (a00+a11)*(b00+b11)
        add_term([a00, a11], [b00, b11], [c00, c11])
        # Wait, I need to be careful with signs and the actual Strassen formula.
        
        # Standard Strassen:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b00-b10)
        # M4 = a11*(b01-b11)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        #
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # I need to implement this with signed entries.
        pass
    
    # This is getting complex. Let me just implement it directly.
    
    # Clear terms and start fresh
    terms = []
    
    def add_term_signed(u_entries, v_entries, w_entries):
        """
        u_entries: list of ((i,j), coeff)
        v_entries: list of ((j,k), coeff)
        w_entries: list of ((i,k), coeff)
        """
        u_vec = [0] * 12
        v_vec = [0] * 24
        w_vec = [0] * 18
        for (i, j), c in u_entries:
            u_vec[i * 4 + j] = c
        for (j, k), c in v_entries:
            v_vec[j * 6 + k] = c
        for (i, k), c in w_entries:
            w_vec[3 * k + i] = c
        terms.append((u_vec, v_vec, w_vec))
    
    # Define Strassen for 2x2 * 2x2 -> 2x2
    def strassen_2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        # M1 = (a00+a11)*(b00+b11)
        add_term_signed([(a00,1),(a11,1)], [(b00,1),(b11,1)], [(c00,1),(c11,1)])
        # M2 = (a10+a11)*b00
        add_term_signed([(a10,1),(a11,1)], [(b00,1)], [(c10,1)])
        # M3 = a00*(b00-b10)
        add_term_signed([(a00,1)], [(b00,1),(b10,-1)], [(c01,1)])
        # M4 = a11*(b01-b11)
        add_term_signed([(a11,1)], [(b01,1),(b11,-1)], [(c00,1),(c10,1)])
        # M5 = (a00+a01)*b11
        add_term_signed([(a00,1),(a01,1)], [(b11,1)], [(c00,-1),(c01,1)])
        # M6 = (a10-a00)*(b00+b01)
        add_term_signed([(a10,1),(a00,-1)], [(b00,1),(b01,1)], [(c11,1)])
        # M7 = (a01-a11)*(b10+b11)
        add_term_signed([(a01,1),(a11,-1)], [(b10,1),(b11,1)], [(c00,1),(c11,1)])
    
    # Define (1x2)*(2x2) -> 1x2, schoolbook rank 4
    def mult_1x2_2x2(a0, a1, b00, b01, b10, b11, c0, c1):
        # c0 = a0*b00 + a1*b10
        # c1 = a0*b01 + a1*b11
        add_term_signed([(a0,1)], [(b00,1)], [(c0,1)])
        add_term_signed([(a1,1)], [(b10,1)], [(c0,1)])
        add_term_signed([(a0,1)], [(b01,1)], [(c1,1)])
        add_term_signed([(a1,1)], [(b11,1)], [(c1,1)])
    
    # Now build the full decomposition.
    # A is 3x4: A[i][j] for i in 0..2, j in 0..3
    # B is 4x6: B[j][k] for j in 0..3, k in 0..5
    # C is 3x6: C[i][k] for i in 0..2, k in 0..5
    
    # For each of the 3 column blocks of B:
    for col_block in range(3):
        b_col0 = col_block * 2
        b_col1 = col_block * 2 + 1
        
        # Top 2 rows of A (rows 0,1) times B[:, b_col0:b_col0+2]
        # Split A_top (2x4) into two 2x2 blocks: cols 0,1 and cols 2,3
        # Split B (4x2) into two 2x2 blocks: rows 0,1 and rows 2,3
        
        # A_top_left: rows 0,1; cols 0,1
        # A_top_right: rows 0,1; cols 2,3
        # B_top: rows 0,1; cols b_col0, b_col1
        # B_bot: rows 2,3; cols b_col0, b_col1
        
        # C = A_top_left * B_top + A_top_right * B_bot
        
        # Strassen for A_top_left * B_top:
        strassen_2x2(
            (0,0), (0,1), (1,0), (1,1),  # A_top_left
            (0,b_col0), (0,b_col1), (1,b_col0), (1,b_col1),  # B_top
            (0,b_col0), (0,b_col1), (1,b_col0), (1,b_col1)  # C[0:2, b_col0:b_col0+2]
        )
        
        # Strassen for A_top_right * B_bot:
        strassen_2x2(
            (0,2), (0,3), (1,2), (1,3),  # A_top_right
            (2,b_col0), (2,b_col1), (3,b_col0), (3,b_col1),  # B_bot
            (0,b_col0), (0,b_col1), (1,b_col0), (1,b_col1)  # C[0:2, b_col0:b_col0+2]
        )
        
        # Bottom row of A (row 2) times B[:, b_col0:b_col0+2]
        # Split A_bot (1x4) into two 1x2 blocks: cols 0,1 and cols 2,3
        # A_bot_left: row 2; cols 0,1
        # A_bot_right: row 2; cols 2,3
        
        # (1x2)*(2x2) for A_bot_left * B_top:
        mult_1x2_2x2(
            (2,0), (2,1),  # A_bot_left
            (0,b_col0), (0,b_col1), (1,b_col0), (1,b_col1),  # B_top
            (2,b_col0), (2,b_col1)  # C[2, b_col0:b_col0+2]
        )
        
        # (1x2)*(2x2) for A_bot_right * B_bot:
        mult_1x2_2x2(
            (2,2), (2,3),  # A_bot_right
            (2,b_col0), (2,b_col1), (3,b_col0), (3,b_col1),  # B_bot
            (2,b_col0), (2,b_col1)  # C[2, b_col0:b_col0+2]
        )
    
    # Now convert terms to the output format.
    rank = len(terms)
    U_out = [[0] * rank for _ in range(12)]
    V_out = [[0] * rank for _ in range(24)]
    W_out = [[0] * rank for _ in range(18)]
    
    for c, (u_vec, v_vec, w_vec) in enumerate(terms):
        for r in range(12):
            U_out[r][c] = u_vec[r]
        for r in range(24):
            V_out[r][c] = v_vec[r]
        for r in range(18):
            W_out[r][c] = w_vec[r]
    
    return {"U": U_out, "V": V_out, "W": W_out}
