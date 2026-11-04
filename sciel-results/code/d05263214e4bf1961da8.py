import numpy as np

def construct():
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Strassen for 4x4 * 4x4
    # Blocks:
    # A00: i in {0,1}, j in {0,1}
    # A01: i in {0,1}, j in {2,3}
    # A10: i in {2,3}, j in {0,1}
    # A11: i in {2,3}, j in {2,3}
    
    # B00: j in {0,1}, k in {0,1}
    # B01: j in {0,1}, k in {2,3}
    # B10: j in {2,3}, k in {0,1}
    # B11: j in {2,3}, k in {2,3}

    # Super Strassen terms M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    # Each Mi is 2x2. We compute Mi using inner Strassen (7 mults).
    # Let's define the 7 inner Strassen terms for generic 2x2 P, Q.
    # P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]]
    # s1 = (p00+p11)(q00+q11)
    # s2 = (p10+p11)q00
    # s3 = p00(q01-q11)
    # s4 = p11(q10-q00)
    # s5 = (p00+p01)q11
    # s6 = (p10-p00)(q00+q01)
    # s7 = (p01-p11)(q10+q11)
    
    # Result S = P Q
    # S00 = s1 + s4 - s5 + s7
    # S01 = s3 + s5
    # S10 = s2 + s4
    # S11 = s1 - s2 + s3 + s6

    # We need to map the "Super" indices to specific linear combinations of A and B elements.
    # For each super term s_idx (0-6), we have a P (linear combo of A blocks) and Q (linear combo of B blocks).
    # Then for each inner term t_idx (0-6), we pick specific elements of P and Q to multiply.

    # Let's define the coefficients for A blocks in P for each super term.
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11

    super_defs = [
        # P_coeffs (A00, A01, A10, A11), Q_coeffs (B00, B01, B10, B11)
        ([1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]),# M3
        ([0, 0, 0, 1], [-1, 0, 1, 0]),# M4
        ([1, 1, 0, 0], [0, 0, 0, 1]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0]),# M6
        ([0, 1, 0, -1], [0, 0, 1, 1]) # M7
    ]

    # Inner Strassen term definitions:
    # Returns (p_elem_indices_with_coeffs, q_elem_indices_with_coeffs, s_contrib_to_S)
    # p_elem is index into P's 2x2 (0:00, 1:01, 2:10, 3:11)
    # q_elem is index into Q's 2x2
    # s_contrib is [S00, S01, S10, S11]

    inner_defs = [
        # t=0: (P00+P11)(Q00+Q11)
        ([(0,1), (3,1)], [(0,1), (3,1)], [1, 0, 0, 1]),
        # t=1: (P10+P11)Q00
        ([(2,1), (3,1)], [(0,1)], [0, 0, 1, -1]),
        # t=2: P00(Q01-Q11)
        ([(0,1)], [(1,1), (3,-1)], [0, 1, 0, 1]),
        # t=3: P11(Q10-Q00)
        ([(3,1)], [(2,1), (0,-1)], [1, 0, 1, 0]),
        # t=4: (P00+P01)Q11
        ([(0,1), (1,1)], [(3,1)], [-1, 1, 0, 0]),
        # t=5: (P10-P00)(Q00+Q01)
        ([(2,1), (0,-1)], [(0,1), (1,1)], [0, 0, 0, 1]),
        # t=6: (P01-P11)(Q10+Q11)
        ([(1,1), (3,-1)], [(2,1), (3,1)], [1, 0, 0, 0])
    ]

    # Helper to expand a block coefficient into individual (i,j) or (j,k) pairs
    def expand_A_block(block_idx, r, c, coeff):
        # block_idx 0: A00, 1: A01, 2: A10, 3: A11
        # r, c are local indices 0..1
        if block_idx == 0:
            return (r, c, coeff)
        elif block_idx == 1:
            return (r, 2+c, coeff)
        elif block_idx == 2:
            return (2+r, c, coeff)
        elif block_idx == 3:
            return (2+r, 2+c, coeff)

    def expand_B_block(block_idx, r, c, coeff):
        # block_idx 0: B00, 1: B01, 2: B10, 3: B11
        # r is local row (j), c is local col (k)
        if block_idx == 0:
            return (r, c, coeff)
        elif block_idx == 1:
            return (r, 2+c, coeff)
        elif block_idx == 2:
            return (2+r, c, coeff)
        elif block_idx == 3:
            return (2+r, 2+c, coeff)

    # Mapping from S element (0:00, 1:01, 2:10, 3:11) and C block (0:00, 1:01, 2:10, 3:11) to global (i,k)
    # C block 0 (C00): i in 0,1; k in 0,1
    # C block 1 (C01): i in 0,1; k in 2,3
    # C block 2 (C10): i in 2,3; k in 0,1
    # C block 3 (C11): i in 2,3; k in 2,3
    
    # Super term s_idx contributes to C blocks with specific signs.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    c_block_contribs = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1],# M2 -> C10, C11
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0],# M5 -> C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]

    for s_idx in range(7):
        p_blocks, q_blocks = super_defs[s_idx]
        c_contribs = c_block_contribs[s_idx]
        
        for t_idx in range(7):
            p_elems, q_elems, s_contrib = inner_defs[t_idx]
            
            u_list = []
            v_list = []
            w_list = []
            
            # Build U (A part)
            for p_elem_idx, p_coeff in p_elems:
                # p_elem_idx 0:00, 1:01, 2:10, 3:11
                r = p_elem_idx // 2
                c = p_elem_idx % 2
                for b_idx, b_coeff in enumerate(p_blocks):
                    if b_coeff == 0: continue
                    i, j, coeff = expand_A_block(b_idx, r, c, b_coeff)
                    u_list.append((i, j, coeff * p_coeff))
            
            # Build V (B part)
            for q_elem_idx, q_coeff in q_elems:
                r = q_elem_idx // 2
                c = q_elem_idx % 2
                for b_idx, b_coeff in enumerate(q_blocks):
                    if b_coeff == 0: continue
                    j, k, coeff = expand_B_block(b_idx, r, c, b_coeff)
                    v_list.append((j, k, coeff * q_coeff))
            
            # Build W (C part)
            for s_pos in range(4):
                s_coeff = s_contrib[s_pos]
                if s_coeff == 0: continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                
                for c_b in range(4):
                    c_coeff = c_contribs[c_b]
                    if c_coeff == 0: continue
                    
                    total_coeff = s_coeff * c_coeff
                    if total_coeff == 0: continue
                    
                    # Map s_r, s_c (local 2x2) and c_b (block) to global i, k
                    if c_b == 0: # C00
                        gi = s_r
                        gk = s_c
                    elif c_b == 1: # C01
                        gi = s_r
                        gk = 2 + s_c
                    elif c_b == 2: # C10
                        gi = 2 + s_r
                        gk = s_c
                    elif c_b == 3: # C11
                        gi = 2 + s_r
                        gk = 2 + s_c
                    
                    w_list.append((gk, gi, total_coeff))
            
            add_term(col, u_list, v_list, w_list)
            col += 1

    # Part 2: Column 4 (Schoolbook)
    # C[i,4] = sum_j A[i,j] B[j,4]
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1

    return {"U": U, "V": V, "W": W}
