import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4, Rank 48 for 4x4x3
    rank = 49 + 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 using Strassen ---
    # Block indices for 2x2 blocks
    # b=0: (0,0), b=1: (0,1), b=2: (1,0), b=3: (1,1)
    
    # Top-level Strassen coefficients for A and B blocks
    # s: 1..7
    # A_blocks[s] = {block_idx: coeff}
    # B_blocks[s] = {block_idx: coeff}
    
    A_top = [
        {}, # 0 unused
        {0: 1, 3: 1}, # 1
        {2: 1, 3: 1}, # 2
        {0: 1},       # 3
        {3: 1},       # 4
        {0: 1, 1: 1}, # 5
        {2: 1, 0: -1},# 6
        {1: 1, 3: -1} # 7
    ]
    
    B_top = [
        {},
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    # Bottom-level Strassen (2x2) coefficients
    # t: 1..7
    # A_blocks[t] = {block_idx: coeff}
    # B_blocks[t] = {block_idx: coeff}
    # C_blocks[t] = {block_idx: coeff} (contribution to C blocks)
    
    # Standard Strassen 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    A_bot = [
        {},
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    B_bot = [
        {},
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    # C contribution for each M_t
    # C_bot[t] = {block_idx: coeff}
    C_bot = [
        {},
        {0: 1, 3: 1}, # M1 contributes to C00 (1) and C11 (1)
        {2: 1, 3: -1}, # M2 contributes to C10 (1) and C11 (-1)
        {1: 1, 3: 1},  # M3 contributes to C01 (1) and C11 (1)
        {0: 1, 2: 1},  # M4 contributes to C00 (1) and C10 (1)
        {0: -1, 1: 1}, # M5 contributes to C00 (-1) and C01 (1)
        {3: 1},        # M6 contributes to C11 (1)
        {0: 1}         # M7 contributes to C00 (1)
    ]
    
    # Helper to map block index to (i_offset, j_offset)
    def block_to_ij(b):
        return (b // 2, b % 2)
    
    col = 0
    for s in range(1, 8):
        for t in range(1, 8):
            # Construct U, V, W for this term
            # U is length 16 (4x4)
            # V is length 28 (4x7), but only first 16 used here
            # W is length 28 (7x4), but only first 16 used here
            
            # Determine effective A coefficients for each element
            # A_eff[i,j] = sum over b_top, b_bot of A_top[s][b_top] * A_bot[t][b_bot] * (coeff of A_ij in block b_bot of A_top[s][b_top] block)
            # This is complex. Let's trace the blocks.
            
            # For a given s, A_top[s] gives coefficients for the 4 blocks of A.
            # Let's call these blocks AB_0, AB_1, AB_2, AB_3.
            # Each AB_b is a 2x2 matrix of original A elements.
            # For a given t, A_bot[t] gives coefficients for the 4 blocks of the "virtual" A matrix formed by s.
            # Let's call the blocks of the virtual A matrix VA_0, VA_1, VA_2, VA_3.
            # VA_b = sum_{b_top} A_top[s][b_top] * AB_{b_top}
            
            # So, for each element A_ij (in block AB_b_top, local pos (li, lj)):
            # It appears in VA_b_bot with coefficient A_top[s][b_top] * 1.
            # Then it is used in M_t with coefficient A_bot[t][b_bot].
            # So total coeff for A_ij is sum_{b_bot} A_top[s][b_top] * A_bot[t][b_bot] where b_top is the block of A_ij.
            
            # Wait, A_top[s] maps original blocks to virtual blocks?
            # No, A_top[s] specifies which original blocks are used to form the virtual blocks.
            # Example s=1: A_top[1] = {0:1, 3:1}.
            # This means Virtual Block 0 = Original Block 0.
            # Virtual Block 3 = Original Block 3.
            # Virtual Block 1, 2 = 0.
            
            # So, for a given original block b_orig:
            # It is part of Virtual Block b_virt if A_top[s] has an entry for b_virt that maps to b_orig?
            # No, the mapping is: Virtual Block b_virt is formed by summing c * Original Block b_orig.
            # In my A_top definition, A_top[s][b_virt] is the coefficient?
            # Let's re-verify A_top definition.
            # A_top[1] = {0: 1, 3: 1}.
            # This means M1 uses (A00+A11).
            # So Virtual Block 0 (top-left of M1's A input) is A00.
            # Virtual Block 3 (bottom-right of M1's A input) is A11.
            # So yes, A_top[s][b_virt] is the coefficient of Original Block b_orig?
            # No, the key is b_orig?
            # In {0: 1, 3: 1}, keys are 0 and 3.
            # If keys are b_orig, then it means Virtual Block 0 gets A00, Virtual Block 3 gets A11.
            # This implies Virtual Block 1 and 2 are 0.
            
            # Let's assume keys in A_top[s] are b_orig, and values are coefficients for the corresponding Virtual Block?
            # No, Strassen formulas are symmetric.
            # M1 = (A00+A11)(B00+B11).
            # A input to M1: Block 0 is A00, Block 3 is A11.
            # So A_top[1] should map Virtual Block 0 -> A00, Virtual Block 3 -> A11.
            # My dict {0: 1, 3: 1} has keys 0, 3.
            # If keys are Virtual Block indices, then:
            # Virt 0 coeff 1 from Orig 0?
            # This is ambiguous.
            
            # Let's define explicitly:
            # For each s, define a 4x4 matrix of coefficients C_A[s][b_virt][b_orig].
            # C_A[s][b_virt][b_orig] is the coefficient of Original Block b_orig in Virtual Block b_virt.
            
            C_A = [[None]*4 for _ in range(8)]
            C_B = [[None]*4 for _ in range(8)]
            for s in range(1, 8):
                for b_virt in range(4):
                    C_A[s][b_virt] = [0]*4
                    C_B[s][b_virt] = [0]*4
            
            # Fill C_A and C_B based on Strassen formulas
            # s=1: A: V0<-O0, V3<-O3. B: V0<-O0, V3<-O3.
            C_A[1][0][0] = 1; C_A[1][3][3] = 1
            C_B[1][0][0] = 1; C_B[1][3][3] = 1
            
            # s=2: A: V2<-O2, V3<-O3. B: V0<-O0.
            C_A[2][2][2] = 1; C_A[2][3][3] = 1
            C_B[2][0][0] = 1
            
            # s=3: A: V0<-O0. B: V1<-O1, V3<-O3(-1).
            C_A[3][0][0] = 1
            C_B[3][1][1] = 1; C_B[3][3][3] = -1
            
            # s=4: A: V3<-O3. B: V2<-O2, V0<-O0(-1).
            C_A[4][3][3] = 1
            C_B[4][2][2] = 1; C_B[4][0][0] = -1
            
            # s=5: A: V0<-O0, V1<-O1. B: V3<-O3.
            C_A[5][0][0] = 1; C_A[5][1][1] = 1
            C_B[5][3][3] = 1
            
            # s=6: A: V2<-O2, V0<-O0(-1). B: V0<-O0, V1<-O1.
            C_A[6][2][2] = 1; C_A[6][0][0] = -1
            C_B[6][0][0] = 1; C_B[6][1][1] = 1
            
            # s=7: A: V1<-O1, V3<-O3(-1). B: V2<-O2, V3<-O3.
            C_A[7][1][1] = 1; C_A[7][3][3] = -1
            C_B[7][2][2] = 1; C_B[7][3][3] = 1
            
            # Now for each s, t:
            for t in range(1, 8):
                # A_bot[t] and B_bot[t] give coefficients for Virtual Blocks in M_t.
                # Let's use the same C_A structure for bottom level?
                # No, bottom level is standard Strassen on 2x2 blocks.
                # A_bot[t][b_virt] is the coefficient of Virtual Block b_virt in M_t's A input.
                # We can define C_A_bot[t][b_virt][b_orig] = A_bot[t][b_virt] if b_orig == b_virt?
                # No, the bottom level operates on the Virtual Blocks directly.
                # So for the bottom level, the "Original" blocks are the Virtual Blocks of the top level.
                
                # So, coefficient of Original Block b_orig in M_t's A input:
                # Sum over b_virt of (Coeff of b_virt in M_t) * (Coeff of b_orig in b_virt from top level)
                # coeff_A[b_orig] = sum_{b_virt} A_bot[t][b_virt] * C_A[s][b_virt][b_orig]
                
                coeff_A = [0]*4
                coeff_B = [0]*4
                for b_orig in range(4):
                    for b_virt in range(4):
                        # A_bot[t] is a dict {b_virt: coeff}
                        if b_virt in A_bot[t]:
                            coeff_A[b_orig] += A_bot[t][b_virt] * C_A[s][b_virt][b_orig]
                        if b_virt in B_bot[t]:
                            coeff_B[b_orig] += B_bot[t][b_virt] * C_B[s][b_virt][b_orig]
                
                # Now coeff_A[b_orig] is the coefficient of Original Block b_orig in M_t.
                # We need to fill U, V, W.
                
                # U: length 16. Index i*4+j.
                # For each b_orig, it corresponds to a 2x2 block.
                # b_orig=0: i in 0,1; j in 0,1.
                # b_orig=1: i in 0,1; j in 2,3.
                # b_orig=2: i in 2,3; j in 0,1.
                # b_orig=3: i in 2,3; j in 2,3.
                
                for b_orig in range(4):
                    if coeff_A[b_orig] != 0:
                        i_off, j_off = block_to_ij(b_orig)
                        for li in range(2):
                            for lj in range(2):
                                i = i_off * 2 + li
                                j = j_off * 2 + lj
                                U[i * 4 + j][col] = coeff_A[b_orig]
                
                # V: length 28. Index j*7+k.
                # We are handling the first 4 columns of B (k in 0..3).
                for b_orig in range(4):
                    if coeff_B[b_orig] != 0:
                        j_off, k_off = block_to_ij(b_orig)
                        for lj in range(2):
                            for lk in range(2):
                                j = j_off * 2 + lj
                                k = k_off * 2 + lk
                                V[j * 7 + k][col] = coeff_B[b_orig]
                
                # W: length 28. Index k*4+i.
                # C_bot[t] gives coefficients for Virtual C Blocks.
                # C_bot[t] is a dict {b_virt_c: coeff}.
                # We need to map Virtual C Blocks to Original C Blocks.
                # The top-level Strassen combines M_s matrices.
                # C_orig[b_orig_c] = sum_s gamma[s][b_orig_c] * M_s[b_orig_c]
                # Wait, M_s is a 2x2 matrix.
                # The contribution of M_s to C_orig[b_orig_c] is M_s[b_orig_c] * gamma[s][b_orig_c].
                # And M_s[b_orig_c] is formed by the bottom-level decomposition.
                # So for a specific b_orig_c, the coefficient from M_s is:
                # sum_{b_virt_c} C_bot[t][b_virt_c] * (1 if b_virt_c == b_orig_c else 0) ?
                # No, the bottom level produces a 2x2 matrix M_s.
                # The element M_s[b_virt_c] is the result of the bottom-level multiplication for that block.
                # So the coefficient of M_s[b_virt_c] in the final sum is C_bot[t][b_virt_c].
                # And the top-level coefficient for M_s[b_orig_c] in C_orig[b_orig_c] is gamma[s][b_orig_c].
                # So total coeff for C_orig[b_orig_c] is:
                # sum_{b_virt_c} C_bot[t][b_virt_c] * gamma[s][b_orig_c] * (1 if b_virt_c == b_orig_c else 0)
                # Wait, M_s is a matrix. M_s[b_virt_c] is an element.
                # The bottom level decomposes the multiplication of two 2x2 matrices into 7 terms.
                # Each term t contributes to the 4 elements of the result matrix.
                # C_bot[t] tells us which elements of the result matrix are affected by term t.
                # So if C_bot[t] has key b_virt_c with value v, it means term t contributes v * (scalar product) to the element b_virt_c of the result matrix M_s.
                # So for the final C_orig[b_orig_c], the contribution is:
                # gamma[s][b_orig_c] * (contribution of term t to M_s[b_orig_c])
                # The contribution of term t to M_s[b_orig_c] is C_bot[t].get(b_orig_c, 0).
                
                # We need gamma[s][b_orig_c].
                # From top-level Strassen:
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                
                gamma = [[0]*4 for _ in range(8)]
                # s=1
                gamma[1][0] = 1; gamma[1][3] = 1
                # s=2
                gamma[2][2] = 1; gamma[2][3] = -1
                # s=3
                gamma[3][1] = 1; gamma[3][3] = 1
                # s=4
                gamma[4][0] = 1; gamma[4][2] = 1
                # s=5
                gamma[5][0] = -1; gamma[5][1] = 1
                # s=6
                gamma[6][3] = 1
                # s=7
                gamma[7][0] = 1
                
                for b_orig_c in range(4):
                    if b_orig_c in C_bot[t]:
                        coeff_W = gamma[s][b_orig_c] * C_bot[t][b_orig_c]
                        if coeff_W != 0:
                            k_off, i_off = block_to_ij(b_orig_c)
                            for lk in range(2):
                                for li in range(2):
                                    k = k_off * 2 + lk
                                    i = i_off * 2 + li
                                    W[k * 4 + i][col] = coeff_W
                                                        
                col += 1
                
    # --- Part 2: 4x4x3 using Schoolbook ---
    # k_global in 4, 5, 6
    for i in range(4):
        for j in range(4):
            for k_local in range(3):
                k_global = 4 + k_local
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
