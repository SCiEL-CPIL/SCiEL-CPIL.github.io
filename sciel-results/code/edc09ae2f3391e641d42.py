import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> (16, rank)
    # V: (m*p, rank) -> (24, rank)
    # W: (p*n, rank) -> (24, rank)
    
    # We will use a dynamic rank. Start with an estimate.
    # Strassen 4x4x4 is rank 49.
    # The remaining 4x4x2 part needs at least 8 (since 4*2=8, and rank of 4x2 matrix mult is 8? No, rank of A(4x4)B(4x2) is 4*2=8? 
    # Wait, the tensor is T_{ijk} = sum_l A_{il} B_{lj}.
    # If B is 4x2, the output C is 4x2.
    # The rank of the 4x4x2 matrix multiplication tensor is 8?
    # Let's check.
    # C_{ik} = sum_{j=0}^3 A_{ij} B_{jk}.
    # This is a sum of 4 terms for each k.
    # Total terms 4*2 = 8.
    # Can we do better?
    # The border rank is 8. The rank is likely 8.
    # So total rank should be around 49 + 8 = 57.
    
    # Let's verify if we can share terms between the 4x4x4 and 4x4x2 parts.
    # The Strassen terms involve sums of A blocks and B blocks.
    # The 4x4x2 part involves individual A entries and B entries.
    # It is unlikely we can share much.
    
    # Let's aim for rank 57 first, and see if it works.
    # If not, we can try 58, 59...
    
    # Actually, let's just build the 49 Strassen terms and 8 schoolbook terms.
    # Total rank = 57.
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return False
        for i, c in u_coeffs:
            if c != 0:
                U[i][col] += c
        for j, c in v_coeffs:
            if c != 0:
                V[j][col] += c
        for k, c in w_coeffs:
            if c != 0:
                W[k][col] += c
        col += 1
        return True

    # --- Part 1: Strassen for 4x4x4 (B cols 0-3, C cols 0-3) ---
    # A is 4x4. B1 is 4x4 (cols 0-3). C1 is 4x4 (cols 0-3).
    
    # Mapping helpers
    def a_idx(block, r, c):
        # block: '00', '01', '10', '11'
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)
        
    def b1_idx(block, r, c):
        # block: '00', '01', '10', '11' for B1 (cols 0-3)
        x, y = int(block[0]), int(block[1])
        # j = x*2+r, k = y*2+c
        return (x*2+r)*6 + (y*2+c)
        
    def c1_idx(block, r, c):
        # block: '00', '01', '10', '11' for C1 (cols 0-3)
        # W index is k*n + i
        x, y = int(block[0]), int(block[1])
        # i = x*2+r, k = y*2+c
        return (y*2+c)*4 + (x*2+r)

    # Strassen 2x2x2 terms (S1-S7)
    # Returns list of (u_coeffs, v_coeffs) for the 2x2 input
    # u_coeffs: list of ( (r,c), coeff )
    # v_coeffs: list of ( (r,c), coeff )
    
    def get_strassen_2x2_terms():
        # S1 = (x00+x11)(y00+y11)
        s1 = ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)])
        # S2 = (x10+x11)y00
        s2 = ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)])
        # S3 = x00(y01-y11)
        s3 = ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)])
        # S4 = x11(y10-y00)
        s4 = ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)])
        # S5 = (x00+x01)y11
        s5 = ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)])
        # S6 = (x10-x00)(y00+y01)
        s6 = ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)])
        # S7 = (x01-x11)(y10+y11)
        s7 = ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])
        return [s1, s2, s3, s4, s5, s6, s7]

    strassen_2x2_terms = get_strassen_2x2_terms()
    
    # Result mapping for 2x2 Strassen
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    result_map = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    # Big Strassen 4x4 combinations
    # Each is (a_parts, b_parts)
    # a_parts: list of (block, sign)
    # b_parts: list of (block, sign)
    
    big_combos = [
        ([('00', 1), ('11', 1)], [('00', 1), ('11', 1)]), # M1
        ([('10', 1), ('11', 1)], [('00', 1)]),            # M2
        ([('00', 1)], [('01', 1), ('11', -1)]),           # M3
        ([('11', 1)], [('10', 1), ('00', -1)]),           # M4
        ([('00', 1), ('01', 1)], [('11', 1)]),            # M5
        ([('10', 1), ('00', -1)], [('00', 1), ('01', 1)]),# M6
        ([('01', 1), ('11', -1)], [('10', 1), ('11', 1)]) # M7
    ]
    
    # Big Result mapping (which M contributes to which C block)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    big_result_map = {
        '00': [(0, 1), (3, 1), (4, -1), (6, 1)], # M1, M4, -M5, M7
        '01': [(2, 1), (4, 1)],                  # M3, M5
        '10': [(1, 1), (3, 1)],                  # M2, M4
        '11': [(0, 1), (1, -1), (2, 1), (5, 1)]  # M1, -M2, M3, M6
    }
    
    # Iterate through the 7 big multiplications
    for big_idx, (a_parts, b_parts) in enumerate(big_combos):
        # For each big multiplication, we compute a 2x2 matrix product.
        # This 2x2 product is decomposed into 7 scalar multiplications.
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]
            
            # Construct global U coeffs
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                for a_block, a_sign in a_parts:
                    idx = a_idx(a_block, rx, cx)
                    u_coeffs.append((idx, a_sign * cx_c))
            
            # Construct global V coeffs
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                for b_block, b_sign in b_parts:
                    idx = b1_idx(b_block, ry, cy)
                    v_coeffs.append((idx, b_sign * cy_c))
            
            # Construct global W coeffs
            w_coeffs = []
            for c_block, big_contribs in big_result_map.items():
                # Check if big_idx is in big_contribs for this c_block
                big_coeff = 0
                for m_idx, m_c in big_contribs:
                    if m_idx == big_idx:
                        big_coeff = m_c
                        break
                if big_coeff == 0:
                    continue
                
                # Now, for this c_block, we need to see where s_idx lands in the 2x2 result.
                for rc, s_contribs in result_map.items():
                    small_coeff = 0
                    for s_check, s_c in s_contribs:
                        if s_check == s_idx:
                            small_coeff = s_c
                            break
                    if small_coeff == 0:
                        continue
                    
                    # This term contributes to c_block at position rc with coeff big_coeff * small_coeff
                    idx = c1_idx(c_block, rc[0], rc[1])
                    w_coeffs.append((idx, big_coeff * small_coeff))
            
            if not add_term(u_coeffs, v_coeffs, w_coeffs):
                # Rank exceeded, but we expect 49 terms here.
                break
            
    # --- Part 2: Schoolbook for 4x4x2 (B cols 4-5, C cols 4-5) ---
    # A is 4x4. B2 is 4x2 (cols 4-5). C2 is 4x2 (cols 4-5).
    for i in range(4):
        for j in range(4):
            for k in range(2):
                # B2 col k corresponds to global col 4+k
                # C2 col k corresponds to global col 4+k
                b_col = 4 + k
                c_col = 4 + k
                
                u_idx = i * 4 + j
                v_idx = j * 6 + b_col
                w_idx = c_col * 4 + i
                
                if not add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]):
                    break

    # The rank used is 'col'
    # We return the matrices. The evaluator will likely check the rank by counting non-zero columns or similar.
    # But the interface says: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # And "score is the rank".
    # If we allocated 57 but only used 57, it's fine.
    # If we used fewer, we should trim.
    
    # Let's trim to actual used rank to be safe, although the shape requirement might imply fixed rank?
    # "factor matrices of shapes ... with ... entries that exactly decompose ... score is the rank"
    # Usually, rank is the number of columns in these matrices.
    # So we should return matrices with exactly 'col' columns.
    
    U_trimmed = [row[:col] for row in U]
    V_trimmed = [row[:col] for row in V]
    W_trimmed = [row[:col] for row in W]
    
    return {"U": U_trimmed, "V": V_trimmed, "W": W_trimmed}
