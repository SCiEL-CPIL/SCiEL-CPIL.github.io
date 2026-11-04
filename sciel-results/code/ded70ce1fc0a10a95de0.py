"""Matrix multiplication 4x4x7 via Strassen-based block decomposition.

Strategy:
- Split B (4x7) into first 4 columns and last 3 columns.
- For 4x4x4 part: use Strassen's algorithm (rank 49).
- For 4x4x3 part: use Strassen 2x2x2 for 4x2 sub-blocks + schoolbook 2x2x1.
  Split B's 3 columns into 2+1.
  - 4x4x2: split into two 4x4x1, each done via 2x2 blocks:
    Actually, let's do 4x4x2 as: split A into 2x2 blocks, B into 2x1 column blocks.
    C = A*B where A is 4x4, B is 4x2.
    C_top = A00*B_top + A01*B_bot (each 2x2 * 2x1)
    C_bot = A10*B_top + A11*B_bot
    Each 2x2*2x1 is rank 4 (schoolbook). So 4x4x2 = 4*4 = 16.
    4x4x3 = 4x4x2 + 4x4x1 = 16 + 16 = 32.
    
  Total: 49 + 32 = 81.

  Wait, can we do better for 4x4x3?
  Let's think of 4x4x3 differently.
  Actually, 4x4x1: A(4x4)*B(4x1) = C(4x1). Rank is at most 16.
  But we can use Strassen-like ideas.
  
  Alternative: Use the fact that 4x4x4 has rank 49 (Strassen).
  For 4x4x3, we could potentially use a subset or modified Strassen.
  
  Let me try: 4x4x3 = split into 4x2x3 + 2x2x3 (row blocks of A).
  4x2x3: A is 4x2, B is 2x3. Rank = 4*2*3 = 24 schoolbook.
  2x2x3: A is 2x2, B is 2x3. Rank = 2*2*3 = 12 schoolbook.
  Total: 24+12 = 36. Worse than 32.
  
  Let's stick with 49 + 32 = 81.
  
  Actually wait - for 4x4x2, can we do better than 16?
  4x4x2: A(4x4)*B(4x2). 
  Split A into 2x2 blocks. B into 2x1 top and 2x1 bottom.
  C_top(2x2) = A00(2x2)*B_top(2x1) + A01(2x2)*B_bot(2x1)
  C_bot(2x2) = A10(2x2)*B_top(2x1) + A11(2x2)*B_bot(2x1)
  
  Each 2x2*2x1 has rank 4. We have 4 such multiplications, so rank 16.
  
  Can we share computations? Not easily with this structure.
  
  So total rank = 49 + 16 + 16 = 81.
"""
import time

def _strassen_2x2x2():
    """
    Returns U (4x7), V (4x7), W (4x7) for 2x2x2 matrix multiplication.
    U index: i*2 + j (i,j in 0..1) -> shape (4, 7)
    V index: j*2 + k (j,k in 0..1) -> shape (4, 7)
    W index: k*2 + i (k,i in 0..1) -> shape (4, 7)
    """
    terms = [
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1, (1,1): -1}},
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1}},
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}},
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}},
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}},
    ]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    for r, t in enumerate(terms):
        for (i, j), val in t['U'].items():
            U[i*2 + j][r] = val
        for (j, k), val in t['V'].items():
            V[j*2 + k][r] = val
        for (k, i), val in t['W'].items():
            W[k*2 + i][r] = val
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Returns U (4x4), V (2x4), W (2x4) for 2x2x1 matrix multiplication.
    A is 2x2, B is 2x1, C is 2x1.
    U index: i*2 + j -> shape (4, 4)
    V index: j*1 + k = j -> shape (2, 4)
    W index: k*2 + i = i -> shape (2, 4)
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    for i in range(2):
        for j in range(2):
            r = i*2 + j
            U[i*2+j][r] = 1
            V[j][r] = 1
            W[i][r] = 1
    return U, V, W

def construct():
    start_time = time.time()
    
    n, m, p = 4, 4, 7
    # Total rank: 49 (for 4x4x4) + 16 (for 4x4x1) + 16 (for 4x4x1) = 81
    # Actually 4x4x3 = 4x4x2 + 4x4x1. 4x4x2 has rank 16, 4x4x1 has rank 16.
    # So total = 49 + 16 + 16 = 81.
    
    rank_total = 49 + 16 + 16
    U = [[0] * rank_total for _ in range(n * m)]  # 16 x 81
    V = [[0] * rank_total for _ in range(m * p)]  # 28 x 81
    W = [[0] * rank_total for _ in range(p * n)]  # 28 x 81
    
    # Part 1: 4x4x4 Strassen (columns 0-3 of B, rows 0-3 of C)
    # Rank offset: 0
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # The 7 super-terms for 4x4x4 Strassen
    super_terms = [
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
    ]
    
    # For each super-term, embed Strassen 2x2x2 for each valid block pair
    rank_offset = 0
    for st in super_terms:
        # For each combination of A-block and B-block where columns of A match rows of B
        for (ai, aj), ca in st['A'].items():
            for (bi, bj), cb in st['B'].items():
                # A block at rows [ai*2:(ai+1)*2], cols [aj*2:(aj+1)*2]
                # B block at rows [bi*2:(bi+1)*2], cols [bj*2:(bj+1)*2]
                # For multiplication to make sense: aj == bi (cols of A = rows of B)
                if aj != bi:
                    continue
                # The product A_block * B_block is a 2x2 matrix
                # C contribution: for each (ci, cj) in C blocks with coeff cc
                # C[ci*2:(ci+1)*2, cj*2:(cj+1)*2] += ca*cb*cc * (A_block * B_block)
                # But we're decomposing A_block * B_block via Strassen 2x2x2
                # The Strassen gives us: sum_r U_s[i,j,r] * V_s[j,k,r] * W_s[k,i,r]
                # where i,j are local to A_block (2x2), j,k local to B_block (2x2), k,i local to C (2x2)
                
                # Embed: A local (i,j) -> global A[(ai*2+i), (aj*2+j)]
                # B local (j,k) -> global B[(bi*2+j), (bj*2+k)] = B[(aj*2+j), (bj*2+k)] since bi=aj
                # C local (k,i) -> global C[(bi*2+k), (ai*2+i)] but C is indexed as C[k_global, i_global]
                # C global row = bi*2+k = aj*2+k, C global col = ai*2+i
                
                for i_loc in range(2):
                    for j_loc in range(2):
                        for r in range(7):
                            val = ca * U_s[i_loc*2+j_loc][r]
                            if val != 0:
                                U[(ai*2+i_loc)*m + (aj*2+j_loc)][rank_offset + r] += val
                for j_loc in range(2):
                    for k_loc in range(2):
                        for r in range(7):
                            val = cb * V_s[j_loc*2+k_loc][r]
                            if val != 0:
                                V[(aj*2+j_loc)*p + (bj*2+k_loc)][rank_offset + r] += val
                # C contribution
                for (ci, cj), cc in st['C'].items():
                    for k_loc in range(2):
                        for i_loc in range(2):
                            for r in range(7):
                                val = ca * cb * cc * W_s[k_loc*2+i_loc][r]
                                if val != 0:
                                    W[(aj*2+k_loc)*n + (ai*2+i_loc)][rank_offset + r] += val
                rank_offset += 7
    
    # rank_offset should be 49 now
    # Part 2: 4x4x1 for column 4 of B (row 4 of C)
    # A(4x4) * B_col4(4x1) = C_row4(4x1)
    # Split A into 2x2 blocks, B into 2x1 top/bottom
    # C_top(2x1) = A00(2x2)*B_top(2x1) + A01(2x2)*B_bot(2x1)
    # C_bot(2x1) = A10(2x2)*B_top(2x1) + A11(2x2)*B_bot(2x1)
    
    U_sb, V_sb, W_sb = _schoolbook_2x2x1()
    
    # 4 sub-multiplications of 2x2x1
    for ai in range(2):
        for aj in range(2):
            # A block at rows [ai*2:(ai+1)*2], cols [aj*2:(aj+1)*2]
            # B: if aj==0, use B_top (rows 0-1); if aj==1, use B_bot (rows 2-3)
            # B global row = aj*2 + j_loc, B col = 4
            # C: if ai==0, use C_top (rows 0-1); if ai==1, use C_bot (rows 2-3)
            # C global row = 4, C global col = ai*2 + i_loc
            
            for i_loc in range(2):
                for j_loc in range(2):
                    r = rank_offset + i_loc*2 + j_loc
                    U[(ai*2+i_loc)*m + (aj*2+j_loc)][r] = 1
                    V[(aj*2+j_loc)*p + 4][r] = 1
                    W[4*n + (ai*2+i_loc)][r] = 1
            rank_offset += 4
    
    # Part 3: 4x4x1 for column 5 of B (row 5 of C)
    for ai in range(2):
        for aj in range(2):
            for i_loc in range(2):
                for j_loc in range(2):
                    r = rank_offset + i_loc*2 + j_loc
                    U[(ai*2+i_loc)*m + (aj*2+j_loc)][r] = 1
                    V[(aj*2+j_loc)*p + 5][r] = 1
                    W[5*n + (ai*2+i_loc)][r] = 1
            rank_offset += 4
    
    # Part 4: 4x4x1 for column 6 of B (row 6 of C)
    for ai in range(2):
        for aj in range(2):
            for i_loc in range(2):
                for j_loc in range(2):
                    r = rank_offset + i_loc*2 + j_loc
                    U[(ai*2+i_loc)*m + (aj*2+j_loc)][r] = 1
                    V[(aj*2+j_loc)*p + 6][r] = 1
                    W[6*n + (ai*2+i_loc)][r] = 1
            rank_offset += 4
    
    return {"U": U, "V": V, "W": W}
