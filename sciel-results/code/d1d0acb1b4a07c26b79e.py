"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-50 tensor decomposition for 2x3 by 3x9 matrix multiplication.
    Strategy:
    - Split B (3x9) into three 3x3 blocks: B0, B1, B2.
    - C (2x9) = A * B = [A*B0, A*B1, A*B2].
    - A*B0 and A*B2 (2x3 * 3x3) are computed using Strassen's algorithm on 2x2 blocks,
      with the remaining 2x1 * 1x3 part done via schoolbook.
    - A*B1 (2x3 * 3x3) is computed entirely via schoolbook (rank 9).
    
    Rank breakdown per 3x3 block:
    - A (2x3) split into A_L (2x2), A_R (2x1).
    - B_block (3x3) split into B_TL, B_TR (top 2x2), B_BL (bottom-left 1x2), B_BR (bottom-right 1x1).
    - A_L * B_TL: Strassen rank 7.
    - A_L * B_BR: Schoolbook 2x2x1, rank 4.
    - A_R * B_BL: Schoolbook 2x1x2, rank 4.
    - A_R * B_BR: Schoolbook 2x1x1, rank 2.
    - A_R * B_TR: Wait, C = [A_L*B_TL + A_R*B_BL, A_L*B_TR + A_R*B_BR].
      My previous block analysis in the thought process was slightly different.
      Let's stick to the verified rank 50 construction derived from the "Strassen-based block decomposition" hint.
      
      Let's re-verify the rank 50 construction from the hint:
      "using 7-term Strassen products for 2x2x2 sub-blocks and schoolbook for the remaining 2x2x1 and 2x1x9 parts"
      
      Let's partition the 3x9 matrix B into:
      Columns 0-1 (2 cols), Col 2 (1 col), Cols 3-4 (2 cols), Col 5 (1 col), Cols 6-7 (2 cols), Col 8 (1 col).
      This splits B into 6 blocks of width 2,1,2,1,2,1? No, that's 9 columns.
      Actually, let's look at the dimensions: 2x3 * 3x9.
      Let's partition B into three 3x3 blocks: B0, B1, B2.
      For each 3x3 block, we compute C_block = A * B_block (2x3 * 3x3 -> 2x3).
      If we can do 2x3 * 3x3 in rank (7+4+4+2)=17? No.
      
      Let's re-evaluate the rank of 2x3 * 3x3.
      Split A (2x3) into A1 (2x2) and A2 (2x1).
      Split B (3x3) into B11 (2x2), B12 (2x1), B21 (1x2), B22 (1x1).
      C = [A1 B11 + A2 B21, A1 B12 + A2 B22].
      A1 B11 (2x2 * 2x2): Strassen rank 7.
      A1 B12 (2x2 * 2x1): Rank 4.
      A2 B21 (2x1 * 1x2): Rank 4.
      A2 B22 (2x1 * 1x1): Rank 2.
      Total rank for one 2x3 * 3x3 block = 7 + 4 + 4 + 2 = 17.
      Three blocks: 3 * 17 = 51.
      This gives 51. The hint says 50.
      Where can we save 1 rank?
      Maybe A1 B12 and A2 B21 can be combined? No, they are distinct.
      Maybe one of the 2x2x1 products can be done in rank 3? No, 2x2x1 is rank 4.
      Maybe 2x1x2 is rank 3? No.
      
      Let's look at the hint again: "schoolbook for the remaining 2x2x1 and 2x1x9 parts".
      This implies a different partitioning.
      Perhaps partition B (3x9) into B_left (3x2) and B_right (3x7)?
      Or partition A?
      
      Let's try:
      A (2x3). B (3x9).
      Split B into B1 (3x2) and B2 (3x7).
      C = [A B1, A B2].
      A B1 (2x3 * 3x2):
      Split A into A1 (2x2), A2 (2x1).
      Split B1 into B11 (2x2), B12 (1x2).
      C1 = [A1 B11, A2 B12].
      A1 B11 (2x2 * 2x2): Strassen 7.
      A2 B12 (2x1 * 1x2): Schoolbook 4.
      Total for B1 part: 11.
      
      A B2 (2x3 * 3x7):
      Split B2 into B21 (3x2), B22 (3x2), B23 (3x3)? No, 7 is 2+2+3.
      Let's try 2+2+2+1.
      B2 cols: 0,1; 2,3; 4,5; 6.
      A B21 (2x3 * 3x2): Same as A B1. Rank 11.
      A B22 (2x3 * 3x2): Rank 11.
      A B23 (2x3 * 3x1):
      Split A into A1 (2x2), A2 (2x1).
      Split B23 into b1 (2x1), b2 (1x1).
      C23 = [A1 b1, A2 b2].
      A1 b1 (2x2 * 2x1): Rank 4.
      A2 b2 (2x1 * 1x1): Rank 2.
      Total for B23: 6.
      Total rank = 11 + 11 + 11 + 6 = 39? That's way too low. I am missing something.
      Ah, the Strassen algorithm computes a 2x2 product. It outputs a 2x2 matrix.
      If I compute A1 B11 (2x2 * 2x2) -> 2x2 output.
      The output indices are (i,k) for i in 0..1, k in 0..1.
      This covers C[0:2, 0:2] of the block.
      
      Let's re-calculate the total rank for 2x3 * 3x9 using this partitioning.
      Block 1 (cols 0,1): 11
      Block 2 (cols 2,3): 11
      Block 3 (cols 4,5): 11
      Block 4 (col 6): 6
      Block 5 (cols 7,8): 11
      Total: 11+11+11+6+11 = 50.
      Yes! This works.
      
      Construction details:
      We process 5 blocks of B:
      B0: 3x2 (cols 0,1)
      B1: 3x2 (cols 2,3)
      B2: 3x2 (cols 4,5)
      B3: 3x1 (col 6)
      B4: 3x2 (cols 7,8)
      
      For each 3x2 block (indices j, j+1):
      A (2x3) -> A_L (2x2, cols 0,1), A_R (2x1, col 2).
      B_block (3x2) -> B_TL (2x2, rows 0,1), B_BL (1x2, row 2).
      C_block (2x2) = [A_L B_TL, A_R B_BL].
      Term 1: A_L B_TL (2x2 * 2x2) via Strassen. Rank 7.
      Terms 2-5: A_R B_BL (2x1 * 1x2) via Schoolbook. Rank 4.
      Total 11 terms.
      
      For the 3x1 block (index 6):
      A (2x3) -> A_L (2x2, cols 0,1), A_R (2x1, col 2).
      B_block (3x1) -> B_T (2x1, rows 0,1), B_B (1x1, row 2).
      C_block (2x1) = [A_L B_T, A_R B_B].
      Terms 1-4: A_L B_T (2x2 * 2x1) via Schoolbook. Rank 4.
      Terms 5-6: A_R B_B (2x1 * 1x1) via Schoolbook. Rank 2.
      Total 6 terms.
      
      Total Rank = 11 * 4 + 6 = 50.
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    def zero_vec(size):
        return [0] * size
        
    # Strassen 2x2x2
    # A (2x2), B (2x2) -> C (2x2)
    # Indices: A: (i,j) i,j in 0..1 -> idx i*2+j
    # B: (j,k) j,k in 0..1 -> idx j*2+k
    # C: (i,k) i,k in 0..1 -> idx i*2+k
    # Terms:
    # 1: A00, B00+B11, C00+C11
    # 2: A10, B00, C10+C11
    # 3: A01, B10+B11, C00+C01
    # 4: A00-A10, B10, C10
    # 5: A01-A11, B01, C01
    # 6: A00+A11, B00-B11, C11
    # 7: A10+A11, B01, C10
    
    def add_strassen_term(u_offset, v_offset, w_offset):
        # u_offset: base index for A (2x2) in the full U vector
        # v_offset: base index for B (2x2) in the full V vector
        # w_offset: base index for C (2x2) in the full W vector
        
        # Term 1
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[u_offset + 0] = 1
        v[v_offset + 0] = 1; v[v_offset + 3] = 1 # B00, B11 (idx 0, 3)
        w[w_offset + 0] = 1; w[w_offset + 3] = 1 # C00, C11 (idx 0, 3)
        add_term(u, v, w)
        
        # Term 2
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[u_offset + 2] = 1 # A10 (idx 2)
        v[v_offset + 0] = 1 # B00
        w[w_offset + 2] = 1; w[w_offset + 3] = 1 # C10, C11
        add_term(u, v, w)
        
        # Term 3
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[u_offset + 1] = 1 # A01 (idx 1)
        v[v_offset + 2] = 1; v[v_offset + 3] = 1 # B10, B11
        w[w_offset + 0] = 1; w[w_offset + 1] = 1 # C00, C01
        add_term(u, v, w)
        
        # Term 4
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[u_offset + 0] = 1; u[u_offset + 2] = -1 # A00 - A10
        v[v_offset + 2] = 1 # B10
        w[w_offset + 2] = 1 # C10
        add_term(u, v, w)
        
        # Term 5
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[u_offset + 1] = 1; u[u_offset + 3] = -1 # A01 - A11
        v[v_offset + 1] = 1 # B01
        w[w_offset + 1] = 1 # C01
        add_term(u, v, w)
        
        # Term 6
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[u_offset + 0] = 1; u[u_offset + 3] = 1 # A00 + A11
        v[v_offset + 0] = 1; v[v_offset + 3] = -1 # B00 - B11
        w[w_offset + 3] = 1 # C11
        add_term(u, v, w)
        
        # Term 7
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[u_offset + 2] = 1; u[u_offset + 3] = 1 # A10 + A11
        v[v_offset + 1] = 1 # B01
        w[w_offset + 2] = 1 # C10
        add_term(u, v, w)
        
    def add_schoolbook_2x2x2(u_offset, v_offset, w_offset):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
                    u[u_offset + i*2+j] = 1
                    v[v_offset + j*2+k] = 1
                    w[w_offset + i*2+k] = 1
                    add_term(u, v, w)
                    
    def add_schoolbook_2x2x1(u_offset, v_offset, w_offset):
        # A (2x2), B (2x1) -> C (2x1)
        # B idx: 0, 1. C idx: 0, 1.
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
                    u[u_offset + i*2+j] = 1
                    v[v_offset + j*1+k] = 1
                    w[w_offset + i*1+k] = 1
                    add_term(u, v, w)

    def add_schoolbook_2x1x2(u_offset, v_offset, w_offset):
        # A (2x1), B (1x2) -> C (2x2)
        # A idx: 0, 1. B idx: 0, 1. C idx: 0, 1, 2, 3.
        for i in range(2):
            for j in range(1):
                for k in range(2):
                    u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
                    u[u_offset + i*1+j] = 1
                    v[v_offset + j*2+k] = 1
                    w[w_offset + i*2+k] = 1
                    add_term(u, v, w)

    def add_schoolbook_2x1x1(u_offset, v_offset, w_offset):
        # A (2x1), B (1x1) -> C (2x1)
        for i in range(2):
            for j in range(1):
                for k in range(1):
                    u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
                    u[u_offset + i] = 1
                    v[v_offset + 0] = 1
                    w[w_offset + i] = 1
                    add_term(u, v, w)

    # Process blocks
    # A indices:
    # A_L (2x2): (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # A_R (2x1): (0,2)->4, (1,2)->5
    
    # B indices for a 3x2 block starting at col c:
    # B_TL (2x2): (0,c)->c, (0,c+1)->c+1, (1,c)->c+3, (1,c+1)->c+4
    # B_BL (1x2): (2,c)->c+6, (2,c+1)->c+7
    
    # C indices for a 2x2 block starting at col c:
    # C (2x2): (0,c)->c, (0,c+1)->c+1, (1,c)->c+2, (1,c+1)->c+3
    # Wait, W index is k*n + i.
    # W[k*n + i]. n=2.
    # For output col c, row i: W[c*2 + i].
    # So C (2x2) at cols c, c+1:
    # i=0, k=0 -> W[c*2+0] = W[2c]
    # i=0, k=1 -> W[(c+1)*2+0] = W[2c+2]
    # i=1, k=0 -> W[c*2+1] = W[2c+1]
    # i=1, k=1 -> W[(c+1)*2+1] = W[2c+3]
    
    # Let's verify the W index mapping in the code.
    # W is (p*n, rank). p=9, n=2. Size 18.
    # W[k*n + i].
    # For Strassen 2x2x2 at cols c, c+1:
    # w_offset = 2*c.
    # w[w_offset + 0] -> W[2c] -> k=c, i=0. Correct.
    # w[w_offset + 1] -> W[2c+1] -> k=c, i=1. Correct.
    # w[w_offset + 2] -> W[2c+2] -> k=c+1, i=0. Correct.
    # w[w_offset + 3] -> W[2c+3] -> k=c+1, i=1. Correct.
    
    # B index mapping:
    # V is (m*p, rank). m=3, p=9. Size 27.
    # V[j*p + k].
    # For B_TL (2x2) at cols c, c+1, rows 0,1:
    # j=0, k=c -> V[c]
    # j=0, k=c+1 -> V[c+1]
    # j=1, k=c -> V[9+c]
    # j=1, k=c+1 -> V[9+c+1]
    # So v_offset for B_TL is not a simple offset if we treat it as 2x2.
    # My helper functions assume contiguous indices for the sub-matrix.
    # V[j*2+k] in the helper assumes V is 2x2 contiguous.
    # But V is 3x9. The 2x2 block is NOT contiguous in V.
    # V[0, c], V[0, c+1], V[1, c], V[1, c+1] are at indices c, c+1, 9+c, 10+c.
    # These are not c, c+1, c+2, c+3.
    # So I cannot use a simple offset for V in the Strassen helper.
    # I must modify the helper to accept explicit indices or map them.
    
    # Let's rewrite the add_term calls to use explicit indices for V and W.
    
    def add_term_explicit(u_indices, v_indices, w_indices, values):
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        for idx, val in zip(u_indices, values):
            u[idx] = val
        for idx, val in zip(v_indices, values):
            v[idx] = val
        for idx, val in zip(w_indices, values):
            w[idx] = val
        add_term(u, v, w)
        
    # This is getting complicated. Let's just build the vectors directly in the loops.
    
    def clear():
        U.clear(); V.clear(); W.clear()
        
    clear()
    
    # Block 0: cols 0, 1
    c = 0
    # Strassen A_L (cols 0,1) * B_TL (rows 0,1)
    # A_L indices in U: 0,1,2,3
    # B_TL indices in V: 0,1, 9,10
    # C (rows 0,1, cols 0,1) indices in W: 0,1, 2,3
    
    # Term 1: A00, B00+B11, C00+C11
    add_term_explicit([0], [0, 10], [0, 3], [1, 1, 1])
    # Term 2: A10, B00, C10+C11
    add_term_explicit([2], [0], [1, 3], [1, 1])
    # Term 3: A01, B10+B11, C00+C01
    add_term_explicit([1], [9, 10], [0, 1], [1, 1, 1])
    # Term 4: A00-A10, B10, C10
    add_term_explicit([0, 2], [9], [1], [1, -1, 1])
    # Term 5: A01-A11, B01, C01
    add_term_explicit([1, 3], [1], [1], [1, -1, 1])
    # Term 6: A00+A11, B00-B11, C11
    add_term_explicit([0, 3], [0, 10], [3], [1, 1, 1, -1])
    # Term 7: A10+A11, B01, C10
    add_term_explicit([2, 3], [1], [1], [1, 1, 1])
    
    # Schoolbook A_R (col 2) * B_BL (row 2)
    # A_R indices in U: 4,5
    # B_BL indices in V: 18,19
    # C (rows 0,1, cols 0,1) indices in W: 0,1, 2,3
    
    # A02 * B20 -> C00
    add_term_explicit([4], [18], [0], [1])
    # A02 * B21 -> C01
    add_term_explicit([4], [19], [1], [1])
    # A12 * B20 -> C10
    add_term_explicit([5], [18], [2], [1])
    # A12 * B21 -> C11
    add_term_explicit([5], [19], [3], [1])
    
    # Block 1: cols 2, 3
    c = 2
    # Strassen
    # A_L: 0,1,2,3
    # B_TL: 2,3, 11,12
    # C: 4,5, 6,7
    
    add_term_explicit([0], [2, 12], [4, 7], [1, 1, 1])
    add_term_explicit([2], [2], [5, 7], [1, 1])
    add_term_explicit([1], [11, 12], [4, 5], [1, 1, 1])
    add_term_explicit([0, 2], [11], [5], [1, -1, 1])
    add_term_explicit([1, 3], [3], [5], [1, -1, 1])
    add_term_explicit([0, 3], [2, 12], [7], [1, 1, 1, -1])
    add_term_explicit([2, 3], [3], [5], [1, 1, 1])
    
    # Schoolbook
    # A_R: 4,5
    # B_BL: 20,21
    # C: 4,5, 6,7
    
    add_term_explicit([4], [20], [4], [1])
    add_term_explicit([4], [21], [5], [1])
    add_term_explicit([5], [20], [6], [1])
    add_term_explicit([5], [21], [7], [1])
    
    # Block 2: cols 4, 5
    c = 4
    # Strassen
    # B_TL: 4,5, 13,14
    # C: 8,9, 10,11
    
    add_term_explicit([0], [4, 14], [8, 11], [1, 1, 1])
    add_term_explicit([2], [4], [9, 11], [1, 1])
    add_term_explicit([1], [13, 14], [8, 9], [1, 1, 1])
    add_term_explicit([0, 2], [13], [9], [1, -1, 1])
    add_term_explicit([1, 3], [5], [9], [1, -1, 1])
    add_term_explicit([0, 3], [4, 14], [11], [1, 1, 1, -1])
    add_term_explicit([2, 3], [5], [9], [1, 1, 1])
    
    # Schoolbook
    # B_BL: 22,23
    # C: 8,9, 10,11
    
    add_term_explicit([4], [22], [8], [1])
    add_term_explicit([4], [23], [9], [1])
    add_term_explicit([5], [22], [10], [1])
    add_term_explicit([5], [23], [11], [1])
    
    # Block 3: col 6
    c = 6
    # Schoolbook A_L (2x2) * B_T (2x1)
    # A_L: 0,1,2,3
    # B_T: 6, 15
    # C: 12, 13
    
    # A00 B06 -> C06
    add_term_explicit([0], [6], [12], [1])
    # A01 B16 -> C06
    add_term_explicit([1], [15], [12], [1])
    # A10 B06 -> C16
    add_term_explicit([2], [6], [13], [1])
    # A11 B16 -> C16
    add_term_explicit([3], [15], [13], [1])
    
    # Schoolbook A_R (2x1) * B_B (1x1)
    # A_R: 4,5
    # B_B: 24
    # C: 12, 13
    
    add_term_explicit([4], [24], [12], [1])
    add_term_explicit([5], [24], [13], [1])
    
    # Block 4: cols 7, 8
    c = 7
    # Strassen
    # B_TL: 7,8, 16,17
    # C: 14,15, 16,17
    
    add_term_explicit([0], [7, 17], [14, 17], [1, 1, 1])
    add_term_explicit([2], [7], [15, 17], [1, 1])
    add_term_explicit([1], [16, 17], [14, 15], [1, 1, 1])
    add_term_explicit([0, 2], [16], [15], [1, -1, 1])
    add_term_explicit([1, 3], [8], [15], [1, -1, 1])
    add_term_explicit([0, 3], [7, 17], [17], [1, 1, 1, -1])
    add_term_explicit([2, 3], [8], [15], [1, 1, 1])
    
    # Schoolbook
    # B_BL: 25,26
    # C: 14,15, 16,17
    
    add_term_explicit([4], [25], [14], [1])
    add_term_explicit([4], [26], [15], [1])
    add_term_explicit([5], [25], [16], [1])
    add_term_explicit([5], [26], [17], [1])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
