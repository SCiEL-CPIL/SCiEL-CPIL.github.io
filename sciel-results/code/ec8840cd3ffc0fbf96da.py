import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 4x4x4 (cols 0-3): 49
    # 4x4x4 (cols 4-7): 49
    # 4x4x1 (col 8): 16
    rank = 49 + 49 + 16
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Strassen 2x2x2 coefficients
    # Returns list of 7 tuples: (list of (a_idx, coeff), list of (b_idx, coeff), list of (c_idx, coeff))
    # a_idx, b_idx, c_idx are 0..3 corresponding to 00, 01, 10, 11
    def get_strassen_222_terms():
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append(([ (0,1), (3,1) ], [ (0,1), (3,1) ], [ (0,1), (1,-1), (2,-1), (3,1) ])) # C00=M1-M4+M5-M7, C01=M3+M7, C10=M2+M6, C11=M1-M3+M2-M6
        # Wait, the C coefficients depend on which M it is.
        # We need to accumulate C contributions for all 7 M's.
        # It's easier to generate the 7 M's, then combine them for C.
        
        # Let's store M terms: (A_coeffs, B_coeffs)
        # M1
        m1_a = [(0,1), (3,1)]
        m1_b = [(0,1), (3,1)]
        # M2
        m2_a = [(2,1), (3,1)]
        m2_b = [(0,1)]
        # M3
        m3_a = [(0,1)]
        m3_b = [(1,1), (3,-1)]
        # M4
        m4_a = [(3,1)]
        m4_b = [(2,1), (0,-1)]
        # M5
        m5_a = [(0,1), (1,1)]
        m5_b = [(3,1)]
        # M6
        m6_a = [(2,1), (0,-1)]
        m6_b = [(0,1), (1,1)]
        # M7
        m7_a = [(1,1), (3,-1)]
        m7_b = [(2,1), (3,1)]
        
        ms = [
            (m1_a, m1_b),
            (m2_a, m2_b),
            (m3_a, m3_b),
            (m4_a, m4_b),
            (m5_a, m5_b),
            (m6_a, m6_b),
            (m7_a, m7_b)
        ]
        return ms

    strassen_222 = get_strassen_222_terms()
    
    # Mapping for 4x4 blocks to global indices
    # Block 00: 0,1,4,5
    # Block 01: 2,3,6,7
    # Block 10: 8,9,12,13
    # Block 11: 10,11,14,15
    block_map = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    def add_strassen_444(offset, b_col_offset):
        # Adds 49 terms for 4x4x4
        # b_col_offset is 0 for cols 0-3, 4 for cols 4-7
        for o in range(7):
            a_coeffs_o, b_coeffs_o = strassen_222[o]
            # For each inner term i
            for i in range(7):
                a_coeffs_i, b_coeffs_i = strassen_222[i]
                
                # Combine outer and inner coefficients
                # This is effectively the Kronecker product of the coefficient vectors
                # But we need to map to global indices
                
                u_entries = {}
                v_entries = {}
                
                # Process A
                for a_block_idx, a_coeff_o in a_coeffs_o:
                    for a_inner_idx, a_coeff_i in a_coeffs_i:
                        # a_block_idx is 0..3 (outer block)
                        # a_inner_idx is 0..3 (inner position in block)
                        # Global index
                        global_a_idx = block_map[a_block_idx][a_inner_idx]
                        coeff = a_coeff_o * a_coeff_i
                        u_entries[global_a_idx] = u_entries.get(global_a_idx, 0) + coeff
                        
                # Process B
                for b_block_idx, b_coeff_o in b_coeffs_o:
                    for b_inner_idx, b_coeff_i in b_coeffs_i:
                        global_b_idx = block_map[b_block_idx][b_inner_idx]
                        # B global row index: j*4 + k
                        # Here j is the outer block row, k is the inner col?
                        # Wait, B is 4x4.
                        # Outer block row r_o (0,1), inner row r_i (0,1).
                        # Outer block col c_o (0,1), inner col c_i (0,1).
                        # block_map index: r_o*2 + c_o.
                        # inner index: r_i*2 + c_i.
                        # Global row: (r_o*2 + r_i)*4 + (c_o*2 + c_i) ?
                        # No, B is 4x4.
                        # Row j, Col k.
                        # j = r_o*2 + r_i.
                        # k = c_o*2 + c_i.
                        # Global index in V: j*4 + k (for 4x4 part).
                        # But V is 4x9. So j*9 + (k + b_col_offset).
                        
                        # My block_map is for 2x2 blocks.
                        # a_block_idx corresponds to (r_o, c_o) of A.
                        # b_block_idx corresponds to (r_o, c_o) of B.
                        # a_inner_idx corresponds to (r_i, c_i) in A's block.
                        # b_inner_idx corresponds to (r_i, c_i) in B's block.
                        
                        # So for B:
                        # r_o = b_block_idx // 2
                        # c_o = b_block_idx % 2
                        # r_i = b_inner_idx // 2
                        # c_i = b_inner_idx % 2
                        
                        j = r_o * 2 + r_i
                        k = c_o * 2 + c_i + b_col_offset
                        global_v_idx = j * 9 + k
                        coeff = b_coeff_o * b_coeff_i
                        v_entries[global_v_idx] = v_entries.get(global_v_idx, 0) + coeff
                        
                # Process C (W)
                # We need to combine the 7 M's to get C.
                # C = sum of M contributions.
                # This logic is inside the o loop? No.
                # The 49 terms are independent rank-1 terms.
                # Each term corresponds to one outer M and one inner M.
                # But C is formed by summing the outer M's.
                # The rank-1 terms are for the products M_{oi}.
                # The output C is linear combination of M_o.
                # So we need to accumulate W contributions for all 7 outer M's?
                # No, the decomposition is a sum of 49 rank-1 tensors.
                # Each rank-1 tensor is $u \otimes v \otimes w$.
                # The $w$ vector for a specific inner term i of outer term o depends on how M_o contributes to C.
                
                # Let's re-evaluate.
                # $C = \sum_{o=1}^7 c_o M_o$.
                # $M_o = \sum_{i=1}^7 u_{oi} v_{oi}^T$ (matrix product).
                # So $C = \sum_{o=1}^7 c_o \sum_{i=1}^7 u_{oi} v_{oi}^T = \sum_{o,i} (c_o u_{oi}) v_{oi}^T$.
                # The rank-1 terms are indexed by (o,i).
                # The U part is $c_o u_{oi}$.
                # The V part is $v_{oi}$.
                # The W part is the identity mapping for the output?
                # No, the output is $C$.
                # The tensor is $C_{jk} = \sum_{i,l} A_{il} B_{lj}$.
                # In the decomposition, $C_{jk} = \sum_{r} U_{?, r} V_{?, r} W_{jk, r}$.
                # The $W$ vector for a term corresponds to the contribution to C.
                # Since $C$ is the sum of $M_o$'s, and each $M_o$ is a $2 \times 2$ matrix.
                # The contribution of $M_{oi}$ to $C$ is $c_o$ times the contribution of the inner $M_i$ to the block $M_o$.
                # The inner $M_i$ is a $2 \times 2$ matrix.
                # So the $W$ vector for term (o,i) is $c_o$ times the $W$ vector of the inner Strassen term i.
                
                # We need the W coefficients for the inner Strassen.
                # C_00 = M1 - M4 + M5 - M7
                # C_01 = M3 + M7
                # C_10 = M2 + M6
                # C_11 = M1 - M3 + M2 - M6
                
                # So for a given outer term o, the coefficient $c_o$ is applied to all 4 blocks of C?
                # No, $M_o$ is a specific block of C?
                # No, $M_o$ is a $2 \times 2$ matrix.
                # $C$ is a $4 \times 4$ matrix composed of 4 $2 \times 2$ blocks.
                # $C_{00} = M_1 - M_4 + M_5 - M_7$ (where $M_k$ are $2 \times 2$ matrices).
                # So the block $C_{00}$ gets contribution from $M_1, M_4, M_5, M_7$.
                # The block $C_{01}$ gets contribution from $M_3, M_7$.
                # etc.
                
                # So for a fixed outer term o, the contribution to C is distributed among the 4 blocks of C.
                # And for each block, the contribution is distributed among the 4 entries of that block (via inner Strassen).
                
                # We need a map: outer_term_o -> { c_block_idx : coeff }
                c_block_coeffs = {
                    0: {0: 1, 3: -1, 4: 1, 6: -1}, # C00
                    1: {2: 1, 6: 1},                 # C01
                    2: {1: 1, 5: 1},                 # C10
                    3: {0: 1, 2: -1, 1: 1, 5: -1}   # C11
                }
                # Wait, the keys in c_block_coeffs are outer term indices (0..6).
                # The values are { c_block_idx : coeff }.
                # But a single outer term o contributes to MULTIPLE c_blocks.
                # So for a fixed o, we iterate over the c_blocks it affects.
                
                # Let's restructure the loop.
                # Loop o in 0..6
                # Loop i in 0..6
                # Loop c_block in 0..3
                # If c_block is affected by o:
                #   Get coeff_c
                #   Get W entries for inner term i, scaled by coeff_c
                
                # This will generate multiple W entries for the same (o,i) term?
                # No, a rank-1 term has a single W vector.
                # The W vector has 16 entries (for 4x4 C).
                # The entries are non-zero only in the blocks affected by o.
                
                # So for each (o,i), we construct the full W vector (16 entries).
                w_entries = {}
                for c_block_idx, coeff_c in c_block_coeffs[o].items():
                    # c_block_idx is 0..3
                    # We need the W coefficients for inner term i.
                    # Inner Strassen C coefficients:
                    # c00: M1 - M4 + M5 - M7
                    # c01: M3 + M7
                    # c10: M2 + M6
                    # c11: M1 - M3 + M2 - M6
                    # Let's define inner_c_coeffs[i] = { c_inner_idx : coeff }
                    
                    # We need a helper to get inner C coeffs for term i.
                    inner_c_coeffs = get_inner_c_coeffs(i)
                    
                    for c_inner_idx, coeff_ic in inner_c_coeffs.items():
                        # c_inner_idx is 0..3 (00, 01, 10, 11)
                        # Global C index
                        r_o = c_block_idx // 2
                        c_o = c_block_idx % 2
                        r_i = c_inner_idx // 2
                        c_i = c_inner_idx % 2
                        k = r_o * 2 + r_i # Row of C
                        l = c_o * 2 + c_i # Col of C
                        global_w_idx = l * 4 + k # W[k*n+i] -> W[l*4+k] ?
                        # Task says W[k*n+i, r]. n=4.
                        # So W index is k*4 + i.
                        # k is col of C, i is row of C.
                        # So global_w_idx = l * 4 + k.
                        
                        coeff_total = coeff_c * coeff_ic
                        w_entries[global_w_idx] = w_entries.get(global_w_idx, 0) + coeff_total

                # Now we have u_entries, v_entries, w_entries for this (o,i) term.
                col = offset + o * 7 + i
                for u_idx, val in u_entries.items():
                    if val != 0:
                        U[u_idx][col] += val
                for v_idx, val in v_entries.items():
                    if val != 0:
                        V[v_idx][col] += val
                for w_idx, val in w_entries.items():
                    if val != 0:
                        W[w_idx][col] += val

    def get_inner_c_coeffs(i):
        # i is 0..6
        # C00: M1(0) - M4(3) + M5(4) - M7(6)
        # C01: M3(2) + M7(6)
        # C10: M2(1) + M6(5)
        # C11: M1(0) - M3(2) + M2(1) - M6(5)
        coeffs = {0:0, 1:0, 2:0, 3:0}
        if i == 0: # M1
            coeffs[0] += 1
            coeffs[3] += 1
        elif i == 1: # M2
            coeffs[2] += 1
            coeffs[3] += 1
        elif i == 2: # M3
            coeffs[1] += 1
            coeffs[3] -= 1
        elif i == 3: # M4
            coeffs[0] -= 1
        elif i == 4: # M5
            coeffs[0] += 1
        elif i == 5: # M6
            coeffs[2] += 1
            coeffs[3] -= 1
        elif i == 6: # M7
            coeffs[0] -= 1
            coeffs[1] += 1
        return coeffs

    def add_schoolbook_441(offset, b_col_idx):
        # Adds 16 terms for 4x4x1
        for i in range(4):
            for j in range(4):
                col = offset + i * 4 + j
                U[i * 4 + j][col] = 1
                V[j * 9 + b_col_idx][col] = 1
                W[b_col_idx * 4 + i][col] = 1

    # Call the builders
    add_strassen_444(0, 0)
    add_strassen_444(49, 4)
    add_schoolbook_441(98, 8)

    return {"U": U, "V": V, "W": W}
