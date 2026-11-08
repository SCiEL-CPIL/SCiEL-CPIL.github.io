"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part, Rank 32 for 4x4x2 part
    rank_strassen = 49
    rank_schoolbook = n * m * (p - 4) # 4 * 4 * 2 = 32
    rank = rank_strassen + rank_schoolbook
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for first 4 columns of B (indices k=0..3) ---
    # A is 4x4, B_part1 is 4x4
    # Block indices for A: (0,0), (0,1), (1,0), (1,1) correspond to 2x2 blocks
    # A_11: rows 0-1, cols 0-1
    # A_12: rows 0-1, cols 2-3
    # A_21: rows 2-3, cols 0-1
    # A_22: rows 2-3, cols 2-3
    
    # Helper to get indices of a 2x2 block in 4x4 matrix
    def block_indices(r, c):
        # r, c in {0, 1}
        rows = [2*r, 2*r+1]
        cols = [2*c, 2*c+1]
        return [(i, j) for i in rows for j in cols]
    
    A_11 = block_indices(0, 0)
    A_12 = block_indices(0, 1)
    A_21 = block_indices(1, 0)
    A_22 = block_indices(1, 1)
    
    B_11 = block_indices(0, 0) # k=0,1
    B_12 = block_indices(0, 1) # k=2,3
    B_21 = block_indices(1, 0) # k=0,1
    B_22 = block_indices(1, 1) # k=2,3
    
    # Define S_k and T_k as linear combinations of A and B blocks
    # S_k is 2x2 matrix of coefficients for A entries
    # T_k is 2x2 matrix of coefficients for B entries
    # Actually, we just need to define the 7 block products P_1..P_7
    # Each P_k = S_k * T_k where S_k, T_k are 2x2 matrices formed from A, B blocks.
    
    # S_1 = A_11 + A_22
    # S_2 = A_21 + A_22
    # S_3 = A_11
    # S_4 = A_22
    # S_5 = A_11 + A_12
    # S_6 = A_21 - A_11
    # S_7 = A_12 - A_22
    
    # T_1 = B_11 + B_22
    # T_2 = B_11
    # T_3 = B_12 - B_22
    # T_4 = B_21 - B_11
    # T_5 = B_22
    # T_6 = B_11 + B_12
    # T_7 = B_21 + B_22
    
    # We represent each S_k and T_k by a list of (index, coeff) pairs for U and V respectively.
    # Since S_k and T_k are 2x2 matrices, we can flatten them.
    # But it's easier to just define the linear combinations directly.
    
    # Let's define the 7 S_k and T_k explicitly as lists of (pos, val)
    # pos is the index in the flattened 4x4 matrix (row * 4 + col)
    
    def make_combo(blocks, coeffs):
        # blocks: list of lists of indices
        # coeffs: list of coefficients corresponding to blocks
        # Returns list of (index, coeff)
        res = []
        for b, c in zip(blocks, coeffs):
            for idx in b:
                res.append((idx, c))
        return res
    
    S_defs = [
        make_combo([A_11, A_22], [1, 1]),
        make_combo([A_21, A_22], [1, 1]),
        make_combo([A_11], [1]),
        make_combo([A_22], [1]),
        make_combo([A_11, A_12], [1, 1]),
        make_combo([A_21, A_11], [1, -1]),
        make_combo([A_12, A_22], [1, -1])
    ]
    
    T_defs = [
        make_combo([B_11, B_22], [1, 1]),
        make_combo([B_11], [1]),
        make_combo([B_12, B_22], [1, -1]),
        make_combo([B_21, B_11], [1, -1]),
        make_combo([B_22], [1]),
        make_combo([B_11, B_12], [1, 1]),
        make_combo([B_21, B_22], [1, 1])
    ]
    
    # Now, for each k in 0..6, we compute P_k = S_k * T_k using Strassen (7 mults)
    # S_k is a 2x2 matrix. Let's denote its entries as s_00, s_01, s_10, s_11.
    # T_k is a 2x2 matrix. Let's denote its entries as t_00, t_01, t_10, t_11.
    # The 7 multiplications for P_k are:
    # m1 = (s_00 + s_11) * (t_00 + t_11)
    # m2 = (s_10 + s_11) * t_00
    # m3 = s_00 * (t_01 - t_11)
    # m4 = s_11 * (t_10 - t_00)
    # m5 = (s_00 + s_01) * t_11
    # m6 = (s_10 - s_00) * (t_00 + t_01)
    # m7 = (s_01 - s_11) * (t_10 + t_11)
    
    # We need to map the entries of S_k and T_k to their linear combinations of A and B.
    # S_k is formed by linear combinations of A blocks.
    # Each entry of S_k (e.g., s_00) is a linear combination of A entries.
    # We can represent s_00 as a dict {index: coeff}.
    
    def block_to_dict(block_indices, coeff):
        d = {}
        for idx in block_indices:
            d[idx] = d.get(idx, 0) + coeff
        return d
    
    def add_dicts(d1, d2):
        d = dict(d1)
        for k, v in d2.items():
            d[k] = d.get(k, 0) + v
            if d[k] == 0:
                del d[k]
        return d
    
    def sub_dicts(d1, d2):
        d = dict(d1)
        for k, v in d2.items():
            d[k] = d.get(k, 0) - v
            if d[k] == 0:
                del d[k]
        return d
        
    # Precompute the 4 entries of each S_k and T_k as dicts
    S_entries = []
    for k in range(7):
        # S_k is a 2x2 matrix. We need to define how each of its 4 entries is formed.
        # The definitions above (S_defs) gave us the whole 2x2 block? No, S_defs gave us the list of (index, coeff) for the whole 2x2 block?
        # Wait, my make_combo flattened the 2x2 block.
        # I need to be more precise.
        # Let's redefine S_k entries.
        pass
        
    # Let's restart the S_k/T_k definition more carefully.
    # A_11 is 2x2. A_11[0,0] is A[0,0], etc.
    # S_1 = A_11 + A_22.
    # S_1[0,0] = A_11[0,0] + A_22[0,0]
    # S_1[0,1] = A_11[0,1] + A_22[0,1]
    # S_1[1,0] = A_11[1,0] + A_22[1,0]
    # S_1[1,1] = A_11[1,1] + A_22[1,1]
    
    def get_block_entry_dict(block, r, c, coeff):
        # block is list of 4 indices [00, 01, 10, 11]
        idx = block[r*2 + c]
        return {idx: coeff}
        
    def sum_dicts(*dicts):
        res = {}
        for d in dicts:
            for k, v in d.items():
                res[k] = res.get(k, 0) + v
                if res[k] == 0:
                    del res[k]
        return res
        
    # Define S_k entries (4 dicts per k)
    S_k_entries = []
    for k in range(7):
        entries = []
        if k == 0: # S_1 = A_11 + A_22
            for r in range(2):
                for c in range(2):
                    entries.append(sum_dicts(get_block_entry_dict(A_11, r, c, 1), get_block_entry_dict(A_22, r, c, 1)))
        elif k == 1: # S_2 = A_21 + A_22
            for r in range(2):
                for c in range(2):
                    entries.append(sum_dicts(get_block_entry_dict(A_21, r, c, 1), get_block_entry_dict(A_22, r, c, 1)))
        elif k == 2: # S_3 = A_11
            for r in range(2):
                for c in range(2):
                    entries.append(get_block_entry_dict(A_11, r, c, 1))
        elif k == 3: # S_4 = A_22
            for r in range(2):
                for c in range(2):
                    entries.append(get_block_entry_dict(A_22, r, c, 1))
        elif k == 4: # S_5 = A_11 + A_12
            for r in range(2):
                for c in range(2):
                    entries.append(sum_dicts(get_block_entry_dict(A_11, r, c, 1), get_block_entry_dict(A_12, r, c, 1)))
        elif k == 5: # S_6 = A_21 - A_11
            for r in range(2):
                for c in range(2):
                    d1 = get_block_entry_dict(A_21, r, c, 1)
                    d2 = get_block_entry_dict(A_11, r, c, -1)
                    entries.append(sum_dicts(d1, d2))
        elif k == 6: # S_7 = A_12 - A_22
            for r in range(2):
                for c in range(2):
                    d1 = get_block_entry_dict(A_12, r, c, 1)
                    d2 = get_block_entry_dict(A_22, r, c, -1)
                    entries.append(sum_dicts(d1, d2))
        S_k_entries.append(entries)
        
    T_k_entries = []
    for k in range(7):
        entries = []
        if k == 0: # T_1 = B_11 + B_22
            for r in range(2):
                for c in range(2):
                    entries.append(sum_dicts(get_block_entry_dict(B_11, r, c, 1), get_block_entry_dict(B_22, r, c, 1)))
        elif k == 1: # T_2 = B_11
            for r in range(2):
                for c in range(2):
                    entries.append(get_block_entry_dict(B_11, r, c, 1))
        elif k == 2: # T_3 = B_12 - B_22
            for r in range(2):
                for c in range(2):
                    d1 = get_block_entry_dict(B_12, r, c, 1)
                    d2 = get_block_entry_dict(B_22, r, c, -1)
                    entries.append(sum_dicts(d1, d2))
        elif k == 3: # T_4 = B_21 - B_11
            for r in range(2):
                for c in range(2):
                    d1 = get_block_entry_dict(B_21, r, c, 1)
                    d2 = get_block_entry_dict(B_11, r, c, -1)
                    entries.append(sum_dicts(d1, d2))
        elif k == 4: # T_5 = B_22
            for r in range(2):
                for c in range(2):
                    entries.append(get_block_entry_dict(B_22, r, c, 1))
        elif k == 5: # T_6 = B_11 + B_12
            for r in range(2):
                for c in range(2):
                    entries.append(sum_dicts(get_block_entry_dict(B_11, r, c, 1), get_block_entry_dict(B_12, r, c, 1)))
        elif k == 6: # T_7 = B_21 + B_22
            for r in range(2):
                for c in range(2):
                    entries.append(sum_dicts(get_block_entry_dict(B_21, r, c, 1), get_block_entry_dict(B_22, r, c, 1)))
        T_k_entries.append(entries)
        
    # Now generate the 49 multiplications
    col = 0
    for k in range(7):
        s00, s01, s10, s11 = S_k_entries[k]
        t00, t01, t10, t11 = T_k_entries[k]
        
        # m1 = (s00 + s11) * (t00 + t11)
        u_dict = sum_dicts(s00, s11)
        v_dict = sum_dicts(t00, t11)
        for idx, val in u_dict.items(): U[idx][col] += val
        for idx, val in v_dict.items(): V[idx][col] += val
        # W part for m1:
        # P_k[0,0] += m1, P_k[1,1] += m1
        # C[0,0] += P_1[0,0] - P_2[0,0] + P_3[0,0] + P_6[0,0]
        # C[0,1] += P_1[0,1] - P_2[0,1] + P_3[0,1] + P_6[0,1]
        # C[1,0] += P_1[1,0] - P_2[1,0] + P_3[1,0] + P_6[1,0]
        # C[1,1] += P_1[1,1] - P_2[1,1] + P_3[1,1] + P_6[1,1]
        # C[0,2] += P_3[0,0] + P_5[0,0] ...
        
        # We need to accumulate W contributions.
        # Let's store the W contributions in a separate structure and add them at the end?
        # Or just add them directly.
        # W index: k_out * 4 + i_out.
        # P_k is a 2x2 block of C.
        # P_k[r,c] contributes to C[2*r + r_out, 2*c + c_out] for the block position.
        # Wait, P_k is the result of multiplying S_k (2x2) and T_k (2x2).
        # The result P_k is a 2x2 matrix.
        # The position of P_k in the 4x4 C matrix depends on k.
        # P_1 is at (0,0), P_2 at (1,0), P_3 at (0,1), P_4 at (1,1)? No.
        # C_11 = P_1 + P_4 - P_5 + P_7
        # C_12 = P_3 + P_5
        # C_21 = P_2 + P_4
        # C_22 = P_1 - P_2 + P_3 + P_6
        
        # So P_k contributes to specific blocks of C.
        # Let's define the coefficients for each P_k in the final C blocks.
        # C_blocks:
        # C_11 (0,0): P_1(1), P_4(1), P_5(-1), P_7(1)
        # C_12 (0,1): P_3(1), P_5(1)
        # C_21 (1,0): P_2(1), P_4(1)
        # C_22 (1,1): P_1(1), P_2(-1), P_3(1), P_6(1)
        
        # For each P_k, we know which C blocks it contributes to and with what coefficient.
        # P_k itself has 4 entries: p00, p01, p10, p11.
        # Each p_rc is a linear combination of the 7 m's.
        # m1 contributes to p00 (+1) and p11 (+1).
        # m2 contributes to p10 (+1) and p11 (-1).
        # m3 contributes to p01 (+1) and p11 (+1).
        # m4 contributes to p00 (+1) and p10 (+1).
        # m5 contributes to p00 (-1) and p01 (+1).
        # m6 contributes to p11 (+1).
        # m7 contributes to p00 (+1).
        
        # So for this specific m (which is m1 for k), we need to add its contribution to W.
        # m1 contributes to P_k[0,0] with +1 and P_k[1,1] with +1.
        # Let's define the contribution of m_j to P_k entries.
        # m1: p00+=1, p11+=1
        # m2: p10+=1, p11-=1
        # m3: p01+=1, p11+=1
        # m4: p00+=1, p10+=1
        # m5: p00-=1, p01+=1
        # m6: p11+=1
        # m7: p00+=1
        
        # We are currently generating m1 for k.
        # So we add to P_k[0,0] and P_k[1,1].
        
        # We will store the W contributions in a list of (w_index, coeff) and add them after all m's are processed?
        # No, we can add them directly to W.
        
        # Function to add contribution of a P_k entry to W
        def add_P_to_W(k, r, c, coeff_m, coeff_P):
            # coeff_P is the coefficient of m in P_k[r,c]
            # We need to find which C blocks P_k contributes to.
            # And for each such block, add to the corresponding W entry.
            # P_k is a 2x2 block.
            # If P_k contributes to C_block(br, bc) with coefficient C_coeff,
            # then P_k[r,c] contributes to C[2*br+r, 2*bc+c] with coefficient C_coeff * coeff_P.
            
            # Get C_coeff for P_k in C_block(br, bc)
            # C_11: k=0(1), k=3(1), k=4(-1), k=6(1)
            # C_12: k=2(1), k=4(1)
            # C_21: k=1(1), k=3(1)
            # C_22: k=0(1), k=1(-1), k=2(1), k=5(1)
            
            contribs = []
            if k == 0: # P_1
                contribs = [(0, 0, 1), (1, 1, 1)] # (br, bc, coeff)
            elif k == 1: # P_2
                contribs = [(1, 0, 1), (1, 1, -1)]
            elif k == 2: # P_3
                contribs = [(0, 1, 1), (1, 1, 1)]
            elif k == 3: # P_4
                contribs = [(0, 0, 1), (1, 0, 1)]
            elif k == 4: # P_5
                contribs = [(0, 0, -1), (0, 1, 1)]
            elif k == 5: # P_6
                contribs = [(1, 1, 1)]
            elif k == 6: # P_7
                contribs = [(0, 0, 1)]
                
            for br, bc, c_coeff in contribs:
                i_out = 2 * br + r
                k_out = 2 * bc + c
                w_idx = k_out * n + i_out
                total_coeff = c_coeff * coeff_P
                if total_coeff != 0:
                    W[w_idx][col] += total_coeff
                    
        # m1: p00 += 1, p11 += 1
        add_P_to_W(k, 0, 0, 1, 1)
        add_P_to_W(k, 1, 1, 1, 1)
        col += 1
        
        # m2 = (s10 + s11) * t00
        u_dict = sum_dicts(s10, s11)
        v_dict = t00
        for idx, val in u_dict.items(): U[idx][col] += val
        for idx, val in v_dict.items(): V[idx][col] += val
        # m2: p10 += 1, p11 -= 1
        add_P_to_W(k, 1, 0, 1, 1)
        add_P_to_W(k, 1, 1, 1, -1)
        col += 1
        
        # m3 = s00 * (t01 - t11)
        u_dict = s00
        v_dict = sum_dicts(t01, {idx: -val for idx, val in t11.items()})
        for idx, val in u_dict.items(): U[idx][col] += val
        for idx, val in v_dict.items(): V[idx][col] += val
        # m3: p01 += 1, p11 += 1
        add_P_to_W(k, 0, 1, 1, 1)
        add_P_to_W(k, 1, 1, 1, 1)
        col += 1
        
        # m4 = s11 * (t10 - t00)
        u_dict = s11
        v_dict = sum_dicts(t10, {idx: -val for idx, val in t00.items()})
        for idx, val in u_dict.items(): U[idx][col] += val
        for idx, val in v_dict.items(): V[idx][col] += val
        # m4: p00 += 1, p10 += 1
        add_P_to_W(k, 0, 0, 1, 1)
        add_P_to_W(k, 1, 0, 1, 1)
        col += 1
        
        # m5 = (s00 + s01) * t11
        u_dict = sum_dicts(s00, s01)
        v_dict = t11
        for idx, val in u_dict.items(): U[idx][col] += val
        for idx, val in v_dict.items(): V[idx][col] += val
        # m5: p00 -= 1, p01 += 1
        add_P_to_W(k, 0, 0, 1, -1)
        add_P_to_W(k, 0, 1, 1, 1)
        col += 1
        
        # m6 = (s10 - s00) * (t00 + t01)
        u_dict = sum_dicts(s10, {idx: -val for idx, val in s00.items()})
        v_dict = sum_dicts(t00, t01)
        for idx, val in u_dict.items(): U[idx][col] += val
        for idx, val in v_dict.items(): V[idx][col] += val
        # m6: p11 += 1
        add_P_to_W(k, 1, 1, 1, 1)
        col += 1
        
        # m7 = (s01 - s11) * (t10 + t11)
        u_dict = sum_dicts(s01, {idx: -val for idx, val in s11.items()})
        v_dict = sum_dicts(t10, t11)
        for idx, val in u_dict.items(): U[idx][col] += val
        for idx, val in v_dict.items(): V[idx][col] += val
        # m7: p00 += 1
        add_P_to_W(k, 0, 0, 1, 1)
        col += 1

    # --- Part 2: Schoolbook for last 2 columns of B (indices k=4,5) ---
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
