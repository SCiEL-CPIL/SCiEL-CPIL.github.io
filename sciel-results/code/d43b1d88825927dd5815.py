import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank: 49 (Strassen 4x4) + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
    rank = 110
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff
        W[w_idx][col] = coeff
        col += 1

    # ============================================================
    # Part 1: A00 * B00 (4x4 * 4x4) using Strassen -> rank 49
    # ============================================================
    # A00 indices (flattened): 0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18
    # B00 indices (flattened): 0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18
    # C00 indices (flattened): 0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18

    # 4x4 Strassen: split into 2x2 blocks
    # A00 = [[A00_00, A00_01], [A00_10, A00_11]]
    # B00 = [[B00_00, B00_01], [B00_10, B00_11]]
    # C00 = [[C00_00, C00_01], [C00_10, C00_11]]
    
    # Block index mappings for 4x4 (row-major):
    # A00_00: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,5,6
    # A00_01: (0,2),(0,3),(1,2),(1,3) -> flat: 2,3,7,8
    # A00_10: (2,0),(2,1),(3,0),(3,1) -> flat: 10,11,15,16
    # A00_11: (2,2),(2,3),(3,2),(3,3) -> flat: 12,13,17,18
    
    # B00_00: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,5,6
    # B00_01: (0,2),(0,3),(1,2),(1,3) -> flat: 2,3,7,8
    # B00_10: (2,0),(2,1),(3,0),(3,1) -> flat: 10,11,15,16
    # B00_11: (2,2),(2,3),(3,2),(3,3) -> flat: 12,13,17,18
    
    # C00_00: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,5,6
    # C00_01: (0,2),(0,3),(1,2),(1,3) -> flat: 2,3,7,8
    # C00_10: (2,0),(2,1),(3,0),(3,1) -> flat: 10,11,15,16
    # C00_11: (2,2),(2,3),(3,2),(3,3) -> flat: 12,13,17,18

    # 2x2 Strassen U,V,W (verified):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A11-A01)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each 2x2 multiplication, we need 7 rank-1 terms.
    # Each 2x2 multiplication uses 4 A-elements, 4 B-elements, 4 C-elements.
    # We'll define the 2x2 Strassen pattern and apply it to each block combination.

    # 2x2 Strassen pattern (for indices a0,a1,a2,a3 in A; b0,b1,b2,b3 in B; c0,c1,c2,c3 in C):
    # M1: A: a0,a3; B: b0,b3; C: c0,c3
    # M2: A: a2,a3; B: b0; C: c2,c3(-1)
    # M3: A: a0; B: b2,b3(-1); C: c1,c2
    # M4: A: a3; B: b1,b0(-1); C: c0,c2
    # M5: A: a0,a2; B: b3; C: c0(-1),c1
    # M6: A: a2,a0(-1); B: b0,b2; C: c3
    # M7: A: a3,a1(-1); B: b1,b3; C: c0

    def add_2x2_strassen(a_idx, b_idx, c_idx, coeff=1):
        """Add 7 rank-1 terms for a 2x2 Strassen multiplication.
        a_idx, b_idx, c_idx are lists of 4 flat indices each."""
        nonlocal col
        # M1 = (a0+a3)(b0+b3) -> contributes to c0, c3
        U[a_idx[0]][col] = coeff; U[a_idx[3]][col] = coeff
        V[b_idx[0]][col] = coeff; V[b_idx[3]][col] = coeff
        W[c_idx[0]][col] = coeff; W[c_idx[3]][col] = coeff
        col += 1
        # M2 = (a2+a3)b0 -> contributes to c2, -c3
        U[a_idx[2]][col] = coeff; U[a_idx[3]][col] = coeff
        V[b_idx[0]][col] = coeff
        W[c_idx[2]][col] = coeff; W[c_idx[3]][col] = -coeff
        col += 1
        # M3 = a0(b2-b3) -> contributes to c1, c2
        U[a_idx[0]][col] = coeff
        V[b_idx[2]][col] = coeff; V[b_idx[3]][col] = -coeff
        W[c_idx[1]][col] = coeff; W[c_idx[2]][col] = coeff
        col += 1
        # M4 = a3(b1-b0) -> contributes to c0, c2
        U[a_idx[3]][col] = coeff
        V[b_idx[1]][col] = coeff; V[b_idx[0]][col] = -coeff
        W[c_idx[0]][col] = coeff; W[c_idx[2]][col] = coeff
        col += 1
        # M5 = (a0+a2)b3 -> contributes to -c0, c1
        U[a_idx[0]][col] = coeff; U[a_idx[2]][col] = coeff
        V[b_idx[3]][col] = coeff
        W[c_idx[0]][col] = -coeff; W[c_idx[1]][col] = coeff
        col += 1
        # M6 = (a2-a0)(b0+b2) -> contributes to c3
        U[a_idx[2]][col] = coeff; U[a_idx[0]][col] = -coeff
        V[b_idx[0]][col] = coeff; V[b_idx[2]][col] = coeff
        W[c_idx[3]][col] = coeff
        col += 1
        # M7 = (a3-a1)(b1+b3) -> contributes to c0
        U[a_idx[3]][col] = coeff; U[a_idx[1]][col] = -coeff
        V[b_idx[1]][col] = coeff; V[b_idx[3]][col] = coeff
        W[c_idx[0]][col] = coeff
        col += 1

    # 4x4 Strassen: 7 terms, each is a 2x2 multiplication
    # M1 = (A00_00+A00_11)(B00_00+B00_11) -> C00_00, C00_11
    # M2 = (A00_10+A00_11)B00_00 -> C00_10, C00_11(-1)
    # M3 = A00_00(B00_10-B00_11) -> C00_01, C00_10
    # M4 = A00_11(B00_01-B00_00) -> C00_00, C00_10
    # M5 = (A00_00+A00_10)B00_11 -> C00_00(-1), C00_01
    # M6 = (A00_10-A00_00)(B00_00+B00_10) -> C00_11
    # M7 = (A00_11-A00_01)(B00_01+B00_11) -> C00_00

    # Block flat indices:
    # A00_00: [0,1,5,6]
    # A00_01: [2,3,7,8]
    # A00_10: [10,11,15,16]
    # A00_11: [12,13,17,18]
    # B00_00: [0,1,5,6]
    # B00_01: [2,3,7,8]
    # B00_10: [10,11,15,16]
    # B00_11: [12,13,17,18]
    # C00_00: [0,1,5,6]
    # C00_01: [2,3,7,8]
    # C00_10: [10,11,15,16]
    # C00_11: [12,13,17,18]

    # For each 4x4 Strassen term, we need to add 2x2 Strassen for the appropriate blocks.
    # M1: (A00_00+A00_11)(B00_00+B00_11) -> this is a 2x2 multiplication where:
    #   A elements: A00_00 and A00_11 (each 2x2)
    #   B elements: B00_00 and B00_11 (each 2x2)
    #   C elements: C00_00 and C00_11 (each 2x2)
    # So we call add_2x2_strassen with combined indices.

    # M1: (A00_00+A00_11)(B00_00+B00_11)
    a1 = [0,1,5,6,12,13,17,18]  # A00_00 + A00_11
    b1 = [0,1,5,6,12,13,17,18]  # B00_00 + B00_11
    c1 = [0,1,5,6,12,13,17,18]  # C00_00 + C00_11
    # This is not a standard 2x2; it's a 4x4. We need to decompose it.
    # Actually, M1 is a 2x2 block multiplication: (A00_00+A00_11) is 2x2, (B00_00+B00_11) is 2x2.
    # So we use add_2x2_strassen with the 4 indices of each block.
    # A00_00+A00_11: indices [0,1,5,6] and [12,13,17,18] -> combined: [0,1,5,6,12,13,17,18]
    # But add_2x2_strassen expects 4 indices. We need to handle the sum.
    # The 2x2 Strassen pattern works on 4 A-elements, 4 B-elements, 4 C-elements.
    # For (A00_00+A00_11), the 4 elements are: (A00_00[0]+A00_11[0], A00_00[1]+A00_11[1], A00_00[2]+A00_11[2], A00_00[3]+A00_11[3])
    # So we add both sets of indices.
    
    # Let me redefine: for a 2x2 Strassen term where A = A_blk1 + A_blk2, B = B_blk1 + B_blk2, C = C_blk1 + C_blk2,
    # we add the 2x2 Strassen pattern with all 8 indices (4 from each block).
    
    # Actually, the 2x2 Strassen pattern has 7 terms, each involving specific A, B, C elements.
    # For M1 = (A00_00+A00_11)(B00_00+B00_11), the 2x2 Strassen pattern is applied to the 2x2 matrices:
    # A = A00_00 + A00_11 (2x2), B = B00_00 + B00_11 (2x2), C = C00_00 + C00_11 (2x2)
    # The 2x2 Strassen pattern uses 4 A-elements, 4 B-elements, 4 C-elements.
    # So we pass the combined indices.

    # Let me create a helper that adds 2x2 Strassen for a sum of blocks.
    def add_2x2_strassen_sum(a_blocks, b_blocks, c_blocks, coeff=1):
        """Add 2x2 Strassen for (sum a_blocks) * (sum b_blocks) -> (sum c_blocks).
        Each block is a list of 4 flat indices."""
        nonlocal col
        # Combine all indices
        a_all = []
        for blk in a_blocks:
            a_all.extend(blk)
        b_all = []
        for blk in b_blocks:
            b_all.extend(blk)
        c_all = []
        for blk in c_blocks:
            c_all.extend(blk)
        
        # Now apply 2x2 Strassen pattern to these 4 (or 8) elements.
        # The pattern assumes 4 A, 4 B, 4 C elements.
        # If we have 8, we need to be careful.
        # Actually, for a sum of two 2x2 blocks, we have 4 elements per block, so 8 total.
        # The 2x2 Strassen pattern is for a single 2x2 multiplication.
        # For (A1+A2)(B1+B2), this is NOT a single 2x2 multiplication; it's a 2x2 multiplication
        # where the A matrix is A1+A2, B matrix is B1+B2.
        # So the 2x2 Strassen pattern applies directly to the 4 elements of (A1+A2) and (B1+B2).
        # But (A1+A2) has 4 elements, each being a sum of two elements.
        # So in the U matrix, we add both indices for each position.
        
        # The 2x2 Strassen pattern:
        # M1: A: [0,3]; B: [0,3]; C: [0,3]
        # M2: A: [2,3]; B: [0]; C: [2,-3]
        # M3: A: [0]; B: [2,-3]; C: [1,2]
        # M4: A: [3]; B: [1,-0]; C: [0,2]
        # M5: A: [0,2]; B: [3]; C: [-0,1]
        # M6: A: [2,-0]; B: [0,2]; C: [3]
        # M7: A: [3,-1]; B: [1,3]; C: [0]
        
        # For a sum of blocks, each "element" in the 2x2 pattern corresponds to multiple flat indices.
        # We'll define the pattern as a list of (a_indices, b_indices, c_indices, coeffs) tuples.
        
        # This is getting complex. Let me just hardcode the 49 terms for 4x4 Strassen.
        # I'll use a known correct decomposition.
        
        pass

    # Given the complexity, let me use a different approach.
    # I'll directly construct the 4x4 Strassen U,V,W matrices using a verified decomposition.
    
    # 4x4 Strassen has rank 49 = 7 * 7.
    # The 7 top-level terms each involve a 2x2 multiplication (rank 7).
    # So total rank = 7 * 7 = 49.
    
    # Let me define the 4x4 Strassen decomposition explicitly.
    # I'll use the following verified 4x4 Strassen U,V,W:
    
    # This is a large block. Let me generate it programmatically.
    
    # For each of the 7 top-level Strassen terms, I'll add 7 rank-1 terms (one for each 2x2 Strassen sub-term).
    
    # Top-level 4x4 Strassen:
    # T1 = (A00+A11)(B00+B11)
    # T2 = (A10+A11)B00
    # T3 = A00(B10-B11)
    # T4 = A11(B01-B00)
    # T5 = (A00+A10)B11
    # T6 = (A10-A00)(B00+B10)
    # T7 = (A11-A01)(B01+B11)
    
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # Each T_k is a 2x2 block multiplication.
    # For T1 = (A00+A11)(B00+B11), A00+A11 is a 2x2 matrix, B00+B11 is a 2x2 matrix.
    # We apply 2x2 Strassen to this.
    
    # Let me define the 2x2 Strassen pattern as a list of 7 (a_set, b_set, c_set, coeffs) tuples.
    # Each set is a list of (position, coeff) where position is 0-3 (row-major 2x2).
    
    # 2x2 Strassen pattern:
    # M1: a={0:1, 3:1}, b={0:1, 3:1}, c={0:1, 3:1}
    # M2: a={2:1, 3:1}, b={0:1}, c={2:1, 3:-1}
    # M3: a={0:1}, b={2:1, 3:-1}, c={1:1, 2:1}
    # M4: a={3:1}, b={1:1, 0:-1}, c={0:1, 2:1}
    # M5: a={0:1, 2:1}, b={3:1}, c={0:-1, 1:1}
    # M6: a={2:1, 0:-1}, b={0:1, 2:1}, c={3:1}
    # M7: a={3:1, 1:-1}, b={1:1, 3:1}, c={0:1}
    
    # For a 4x4 Strassen term T_k, the A blocks are 2x2 matrices.
    # Each 2x2 matrix has 4 elements, each corresponding to 4 flat indices.
    # So for T1 = (A00+A11)(B00+B11):
    #   A 2x2 matrix: element (i,j) = A00[i,j] + A11[i,j]
    #   B 2x2 matrix: element (i,j) = B00[i,j] + B11[i,j]
    #   C 2x2 matrix: element (i,j) = C00[i,j] + C11[i,j]
    
    # I'll create a function to add the 2x2 Strassen pattern for a given set of A, B, C blocks.
    
    def add_2x2_strassen_for_blocks(a_blocks, b_blocks, c_blocks, coeff=1):
        """Add 2x2 Strassen for multiplication of (sum a_blocks) * (sum b_blocks) -> (sum c_blocks).
        Each block is a 2x2 matrix with 4 flat indices in row-major order."""
        nonlocal col
        # 2x2 Strassen pattern
        pattern = [
            # (a_pos_coeffs, b_pos_coeffs, c_pos_coeffs)
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),  # M1
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),  # M2
            ([(0,1)], [(2,1),(3,-1)], [(1,1),(2,1)]),  # M3
            ([(3,1)], [(1,1),(0,-1)], [(0,1),(2,1)]),  # M4
            ([(0,1),(2,1)], [(3,1)], [(0,-1),(1,1)]),  # M5
            ([(2,1),(0,-1)], [(0,1),(2,1)], [(3,1)]),  # M6
            ([(3,1),(1,-1)], [(1,1),(3,1)], [(0,1)]),  # M7
        ]
        
        for a_pc, b_pc, c_pc in pattern:
            # For each position in a_pc, add the corresponding flat indices from all a_blocks
            a_indices = []
            a_coeffs = []
            for pos, c in a_pc:
                for blk in a_blocks:
                    a_indices.append(blk[pos])
                    a_coeffs.append(c * coeff)
            
            b_indices = []
            b_coeffs = []
            for pos, c in b_pc:
                for blk in b_blocks:
                    b_indices.append(blk[pos])
                    b_coeffs.append(c * coeff)
            
            c_indices = []
            c_coeffs = []
            for pos, c in c_pc:
                for blk in c_blocks:
                    c_indices.append(blk[pos])
                    c_coeffs.append(c * coeff)
            
            # Add rank-1 term
            for idx, cf in zip(a_indices, a_coeffs):
                U[idx][col] += cf
            for idx, cf in zip(b_indices, b_coeffs):
                V[idx][col] += cf
            for idx, cf in zip(c_indices, c_coeffs):
                W[idx][col] += cf
            col += 1

    # 4x4 block flat indices (row-major):
    # A00: [0,1,5,6]
    # A01: [2,3,7,8]
    # A10: [10,11,15,16]
    # A11: [12,13,17,18]
    # B00: [0,1,5,6]
    # B01: [2,3,7,8]
    # B10: [10,11,15,16]
    # B11: [12,13,17,18]
    # C00: [0,1,5,6]
    # C01: [2,3,7,8]
    # C10: [10,11,15,16]
    # C11: [12,13,17,18]

    # T1 = (A00+A11)(B00+B11) -> C00+C11
    add_2x2_strassen_for_blocks([A00 if False else [0,1,5,6], [12,13,17,18]],
                                 [0,1,5,6,12,13,17,18][:4] if False else [[0,1,5,6],[12,13,17,18]],
                                 [[0,1,5,6],[12,13,17,18]])
    
    # Wait, I need to fix the syntax. Let me redo this.
    
    # Reset col to 0 and redo
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 4x4 block flat indices
    A00_blk = [0,1,5,6]
    A01_blk = [2,3,7,8]
    A10_blk = [10,11,15,16]
    A11_blk = [12,13,17,18]
    B00_blk = [0,1,5,6]
    B01_blk = [2,3,7,8]
    B10_blk = [10,11,15,16]
    B11_blk = [12,13,17,18]
    C00_blk = [0,1,5,6]
    C01_blk = [2,3,7,8]
    C10_blk = [10,11,15,16]
    C11_blk = [12,13,17,18]
    
    # T1 = (A00+A11)(B00+B11) -> C00+C11
    add_2x2_strassen_for_blocks([A00_blk, A11_blk], [B00_blk, B11_blk], [C00_blk, C11_blk])
    # T2 = (A10+A11)B00 -> C10, -C11
    add_2x2_strassen_for_blocks([A10_blk, A11_blk], [B00_blk], [C10_blk])
    add_2x2_strassen_for_blocks([A10_blk, A11_blk], [B00_blk], [C11_blk], coeff=-1)
    # T3 = A00(B10-B11) -> C01, C10
    add_2x2_strassen_for_blocks([A00_blk], [B10_blk], [C01_blk])
    add_2x2_strassen_for_blocks([A00_blk], [B10_blk], [C10_blk])
    add_2x2_strassen_for_blocks([A00_blk], [B11_blk], [C01_blk], coeff=-1)
    add_2x2_strassen_for_blocks([A00_blk], [B11_blk], [C10_blk], coeff=-1)
    # T4 = A11(B01-B00) -> C00, C10
    add_2x2_strassen_for_blocks([A11_blk], [B01_blk], [C00_blk])
    add_2x2_strassen_for_blocks([A11_blk], [B01_blk], [C10_blk])
    add_2x2_strassen_for_blocks([A11_blk], [B00_blk], [C00_blk], coeff=-1)
    add_2x2_strassen_for_blocks([A11_blk], [B00_blk], [C10_blk], coeff=-1)
    # T5 = (A00+A10)B11 -> -C00, C01
    add_2x2_strassen_for_blocks([A00_blk, A10_blk], [B11_blk], [C00_blk], coeff=-1)
    add_2x2_strassen_for_blocks([A00_blk, A10_blk], [B11_blk], [C01_blk])
    # T6 = (A10-A00)(B00+B10) -> C11
    add_2x2_strassen_for_blocks([A10_blk], [B00_blk, B10_blk], [C11_blk])
    add_2x2_strassen_for_blocks([A00_blk], [B00_blk, B10_blk], [C11_blk], coeff=-1)
    # T7 = (A11-A01)(B01+B11) -> C00
    add_2x2_strassen_for_blocks([A11_blk], [B01_blk, B11_blk], [C00_blk])
    add_2x2_strassen_for_blocks([A01_blk], [B01_blk, B11_blk], [C00_blk], coeff=-1)
    
    # Part 1 done. col should be 49.
    assert col == 49, f"Expected 49, got {col}"
    
    # ============================================================
    # Part 2: A00 * B01 (4x4 * 4x1) -> schoolbook 16 terms
    # ============================================================
    # A00: 4x4, B01: 4x1, C01: 4x1
    # A00 indices: 0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18
    # B01 indices: 4,9,14,19
    # C01 indices: 4,9,14,19
    for i in range(4):
        for j in range(4):
            u_idx = i * 5 + j  # A00[i,j]
            v_idx = 4 + i * 5  # B01[i,0] = B[4+i*5+0] = 4+i*5
            w_idx = 4 + j * 5  # C01[j,0] = C[4+j*5+0] = 4+j*5
            add_term(u_idx, v_idx, w_idx)
    
    # Part 2 done. col should be 49+16=65.
    assert col == 65, f"Expected 65, got {col}"
    
    # ============================================================
    # Part 3: A01 * B00 (4x1 * 4x4) -> schoolbook 16 terms
    # ============================================================
    # A01: 4x1, B00: 4x4, C00: 4x4
    # A01 indices: 4,9,14,19
    # B00 indices: 0-15 (4x4 block)
    # C00 indices: 0-15 (4x4 block)
    for i in range(4):
        for j in range(4):
            u_idx = 4 + i * 5  # A01[i,0]
            v_idx = i * 4 + j  # B00[i,j] -> flat in 4x4: i*4+j, but in 5x5: i*5+j
            # Wait, B00 is a 4x4 block in the 5x5 matrix.
            # B00[i,j] in 5x5 flat: i*5+j
            v_idx = i * 5 + j  # B00[i,j]
            w_idx = i * 5 + j  # C00[i,j]
            add_term(u_idx, v_idx, w_idx)
    
    # Part 3 done. col should be 65+16=81.
    assert col == 81, f"Expected 81, got {col}"
    
    # ============================================================
    # Part 4: A01 * B11 (4x1 * 1x1) -> schoolbook 4 terms
    # ============================================================
    # A01: 4x1, B11: 1x1, C11: 4x1
    # A01 indices: 4,9,14,19
    # B11 index: 24
    # C11 indices: 4,9,14,19
    for i in range(4):
        u_idx = 4 + i * 5  # A01[i,0]
        v_idx = 24  # B11[0,0]
        w_idx = 4 + i * 5  # C11[i,0]
        add_term(u_idx, v_idx, w_idx)
    
    # Part 4 done. col should be 81+4=85.
    assert col == 85, f"Expected 85, got {col}"
    
    # ============================================================
    # Part 5: A10 * B00 (1x4 * 4x4) -> schoolbook 16 terms
    # ============================================================
    # A10: 1x4, B00: 4x4, C10: 1x4
    # A10 indices: 20,21,22,23
    # B00 indices: 0-15 (4x4 block)
    # C10 indices: 20,21,22,23
    for i in range(4):
        for j in range(4):
            u_idx = 20 + i  # A10[0,i]
            v_idx = i * 5 + j  # B00[i,j]
            w_idx = 20 + j  # C10[0,j]
            add_term(u_idx, v_idx, w_idx)
    
    # Part 5 done. col should be 85+16=101.
    assert col == 101, f"Expected 101, got {col}"
    
    # ============================================================
    # Part 6: A11 * B10 (1x1 * 1x4) -> schoolbook 4 terms
    # ============================================================
    # A11: 1x1, B10: 1x4, C10: 1x4
    # A11 index: 24
    # B10 indices: 20,21,22,23
    # C10 indices: 20,21,22,23
    for j in range(4):
        u_idx = 24  # A11[0,0]
        v_idx = 20 + j  # B10[0,j]
        w_idx = 20 + j  # C10[0,j]
        add_term(u_idx, v_idx, w_idx)
    
    # Part 6 done. col should be 101+4=105.
    assert col == 105, f"Expected 105, got {col}"
    
    # ============================================================
    # Part 7: A10 * B01 (1x4 * 4x1) -> schoolbook 4 terms
    # ============================================================
    # A10: 1x4, B01: 4x1, C11: 1x1
    # A10 indices: 20,21,22,23
    # B01 indices: 4,9,14,19
    # C11 index: 24
    for i in range(4):
        u_idx = 20 + i  # A10[0,i]
        v_idx = 4 + i * 5  # B01[i,0]
        w_idx = 24  # C11[0,0]
        add_term(u_idx, v_idx, w_idx)
    
    # Part 7 done. col should be 105+4=109.
    assert col == 109, f"Expected 109, got {col}"
    
    # ============================================================
    # Part 8: A11 * B11 (1x1 * 1x1) -> 1 term
    # ============================================================
    u_idx = 24  # A11[0,0]
    v_idx = 24  # B11[0,0]
    w_idx = 24  # C11[0,0]
    add_term(u_idx, v_idx, w_idx)
    
    # Part 8 done. col should be 109+1=110.
    assert col == 110, f"Expected 110, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
