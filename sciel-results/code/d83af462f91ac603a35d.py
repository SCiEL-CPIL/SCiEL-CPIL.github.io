"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 2x4x6 part: 6 blocks of 2x2x2 (rank 7) -> 42
    # 2x1x6 part: 1 block of rank 1 -> 1
    # Total rank = 43
    rank = 43
    
    U = [[0] * rank for _ in range(n * m)] # 10 x 43
    V = [[0] * rank for _ in range(m * p)] # 30 x 43
    W = [[0] * rank for _ in range(p * n)] # 12 x 43
    
    # Strassen coefficients for 2x2x2
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Note: W index is k*n+i.
    # Local W indices:
    # k=0,i=0 -> 0
    # k=1,i=0 -> 1
    # k=0,i=1 -> 2
    # k=1,i=1 -> 3
    
    # Term 1: M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # U: 0,3. V: 0,3. W: 0,3.
    s1 = {
        'U': {0: 1, 3: 1},
        'V': {0: 1, 3: 1},
        'W': {0: 1, 3: 1}
    }
    
    # Term 2: M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
    # Local C indices: C10 is (i=1, k=0) -> k*n+i = 0*2+1 = 1. C11 is (i=1, k=1) -> 1*2+1 = 3.
    s2 = {
        'U': {2: 1, 3: 1},
        'V': {0: 1},
        'W': {1: 1, 3: -1}
    }
    
    # Term 3: M3 = A00(B01-B11) -> C01+=M3
    # Local C index: C01 is (i=0, k=1) -> 1*2+0 = 2.
    s3 = {
        'U': {0: 1},
        'V': {1: 1, 3: -1},
        'W': {2: 1}
    }
    
    # Term 4: M4 = A11(B10-B00) -> C10+=M4, C11+=M4
    # Local C indices: C10 -> 1, C11 -> 3.
    s4 = {
        'U': {3: 1},
        'V': {2: 1, 0: -1},
        'W': {1: 1, 3: 1}
    }
    
    # Term 5: M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
    # Local C indices: C00 -> 0, C01 -> 2.
    s5 = {
        'U': {0: 1, 1: 1},
        'V': {3: 1},
        'W': {0: -1, 2: 1}
    }
    
    # Term 6: M6 = (A10-A00)(B00+B01) -> C11+=M6
    # Local C index: C11 -> 3.
    s6 = {
        'U': {2: 1, 0: -1},
        'V': {0: 1, 1: 1},
        'W': {3: 1}
    }
    
    # Term 7: M7 = (A01-A11)(B10+B11) -> C00+=M7, C10-=M7
    # Local C indices: C00 -> 0, C10 -> 1.
    s7 = {
        'U': {1: 1, 3: -1},
        'V': {2: 1, 3: 1},
        'W': {0: 1, 1: -1}
    }
    
    strassen_terms = [s1, s2, s3, s4, s5, s6, s7]
    
    def add_strassen_block(offset, j_offset, k_offset):
        # j_offset: 0 or 2 (column in A, row in B)
        # k_offset: 0, 2, or 4 (column in B)
        for t_idx, term in enumerate(strassen_terms):
            r = offset + t_idx
            # Fill U
            for local_u, val in term['U'].items():
                # local_u is 0..3.
                # local_u = i*2 + j? No, standard row major: i*2+j.
                # 0:00, 1:01, 2:10, 3:11
                i = local_u // 2
                j = local_u % 2
                global_u_idx = i * m + (j + j_offset)
                U[global_u_idx][r] = val
                
            # Fill V
            for local_v, val in term['V'].items():
                # local_v = j*2 + k?
                # 0:00, 1:01, 2:10, 3:11
                j = local_v // 2
                k = local_v % 2
                global_v_idx = (j + j_offset) * p + (k + k_offset)
                V[global_v_idx][r] = val
                
            # Fill W
            for local_w, val in term['W'].items():
                # local_w is index in C (2x2).
                # C local index: k*2 + i?
                # My previous mapping:
                # C00 (i=0,k=0) -> 0
                # C01 (i=0,k=1) -> 1? No.
                # Task convention: W[k*n+i, r].
                # Local n=2.
                # C00: k=0, i=0 -> 0*2+0 = 0.
                # C01: k=1, i=0 -> 1*2+0 = 2.
                # C10: k=0, i=1 -> 0*2+1 = 1.
                # C11: k=1, i=1 -> 1*2+1 = 3.
                
                # Let's re-verify the Strassen W mappings with this convention.
                # s1: C00, C11. Indices 0, 3. Correct.
                # s2: C10, C11. Indices 1, 3. Correct.
                # s3: C01. Index 2. Correct.
                # s4: C10, C11. Indices 1, 3. Correct.
                # s5: C00, C01. Indices 0, 2. Correct.
                # s6: C11. Index 3. Correct.
                # s7: C00, C10. Indices 0, 1. Correct.
                
                k = local_w // 2
                i = local_w % 2
                global_w_idx = (k + k_offset) * n + i
                W[global_w_idx][r] = val

    # Add 6 blocks for 2x4x6
    # Block 1: A cols 0,1. B rows 0,1. B cols 0,1.
    add_strassen_block(0, 0, 0)
    # Block 2: A cols 0,1. B rows 0,1. B cols 2,3.
    add_strassen_block(7, 0, 2)
    # Block 3: A cols 0,1. B rows 0,1. B cols 4,5.
    add_strassen_block(14, 0, 4)
    # Block 4: A cols 2,3. B rows 2,3. B cols 0,1.
    add_strassen_block(21, 2, 0)
    # Block 5: A cols 2,3. B rows 2,3. B cols 2,3.
    add_strassen_block(28, 2, 2)
    # Block 6: A cols 2,3. B rows 2,3. B cols 4,5.
    add_strassen_block(35, 2, 4)
    
    # Add 1 block for 2x1x6
    # A col 4. B row 4.
    # Rank 1.
    r = 42
    # U: i in 0,1. j=4.
    U[0*5 + 4][r] = 1
    U[1*5 + 4][r] = 1
    
    # V: j=4. k in 0..5.
    for k in range(6):
        V[4*6 + k][r] = 1
        
    # W: k in 0..5. i in 0,1.
    for k in range(6):
        for i in range(2):
            W[k*2 + i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
