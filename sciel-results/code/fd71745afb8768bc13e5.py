import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part (Strassen) + 48 for 4x4x3 part (Schoolbook)
    rank = 49 + 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get Strassen 2x2x2 coefficients
    # Returns list of 7 tuples (U_flat, V_flat, W_flat)
    # U_flat, V_flat, W_flat are length 4
    def get_strassen_2x2():
        # s=1
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1] # C00, C11
        # s=2
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 1, 0, -1] # C10, C11(-1)
        # s=3
        u3 = [1, 0, 0, 0]
        v3 = [0, 1, 0, -1]
        w3 = [0, 0, 1, 1] # C01, C11
        # s=4
        u4 = [0, 0, 0, 1]
        v4 = [-1, 0, 1, 0]
        w4 = [1, 1, 0, 0] # C00, C10
        # s=5
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [-1, 0, 1, 0] # C00(-1), C01
        # s=6
        u6 = [-1, 0, 1, 0]
        v6 = [1, 1, 0, 0]
        w6 = [0, 0, 0, 1] # C11
        # s=7
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        w7 = [1, 0, 0, 0] # C00
        
        return [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]

    strassen_coeffs = get_strassen_2x2()
    
    # Outer Strassen forms for 4x4
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # We need to combine the outer linear forms with the inner Strassen forms.
    # For each outer product s (1..7), we have a linear form in A blocks and B blocks.
    # Let LA_s = sum alpha_ab A_ab
    # Let LB_s = sum beta_cd B_cd
    # Then M_s = LA_s LB_s
    # M_s is a 2x2 matrix product. We decompose it using inner Strassen.
    # M_s = sum_t (L^A_{s,t}) (L^B_{s,t})
    # where L^A_{s,t} is a linear form in A elements, L^B_{s,t} in B elements.
    
    # Outer coefficients for A blocks (00, 01, 10, 11)
    outer_A = [
        [1, 0, 0, 1], # s1: A00 + A11
        [0, 0, 1, 1], # s2: A10 + A11
        [1, 0, 0, 0], # s3: A00
        [0, 0, 0, 1], # s4: A11
        [1, 1, 0, 0], # s5: A00 + A01
        [-1, 0, 1, 0],# s6: A10 - A00
        [0, 1, 0, -1] # s7: A01 - A11
    ]
    
    # Outer coefficients for B blocks (00, 01, 10, 11)
    outer_B = [
        [1, 0, 0, 1], # s1: B00 + B11
        [1, 0, 0, 0], # s2: B00
        [0, 1, 0, -1],# s3: B01 - B11
        [0, 0, 1, -1],# s4: B10 - B00
        [0, 0, 0, 1], # s5: B11
        [1, 1, 0, 0], # s6: B00 + B01
        [0, 0, 1, 1]  # s7: B10 + B11
    ]
    
    # Outer coefficients for C blocks (00, 01, 10, 11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    outer_C = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2 (C10=1, C11=-1) -> Wait, W index is k*n+i.
        # C10 is i=1, k=0. Index 0*4+1 = 1.
        # C11 is i=1, k=1. Index 1*4+1 = 5.
        # So W_flat for 4x4 (size 16) needs to be careful.
        # Let's stick to the 2x2 inner W and map it.
    ]
    
    # Actually, it's easier to iterate over s and t.
    # For each s, t:
    # U_ij is sum over A_blocks ab: outer_A[s][ab] * inner_U[t][local_ij]
    # V_jk is sum over B_blocks cd: outer_B[s][cd] * inner_V[t][local_jk]
    # W_ki is sum over C_blocks ef: outer_C[s][ef] * inner_W[t][local_ki]
    
    # Let's precompute inner_U, inner_V, inner_W as 7x4x4 matrices?
    # inner_U[t][block_pos][local_idx]
    
    col = 0
    for s in range(7):
        for t in range(7):
            u_inner = strassen_coeffs[t][0]
            v_inner = strassen_coeffs[t][1]
            w_inner = strassen_coeffs[t][2]
            
            # Construct U for this column
            # U has size 16 (4x4)
            # u_inner has size 4 (2x2)
            # outer_A[s] has size 4 (coeffs for A00, A01, A10, A11)
            for ab_idx, coeff in enumerate(outer_A[s]):
                if coeff == 0: continue
                # ab_idx: 0->A00, 1->A01, 2->A10, 3->A11
                # Block row: ab_idx // 2
                # Block col: ab_idx % 2
                # Local indices in block: 0..3
                for local_idx, val in enumerate(u_inner):
                    if val == 0: continue
                    # Map global A index
                    # A is 4x4.
                    # Block (br, bc) covers rows 2*br, 2*br+1 and cols 2*bc, 2*bc+1.
                    # local_idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                    li = local_idx // 2
                    lj = local_idx % 2
                    global_i = 2 * (ab_idx // 2) + li
                    global_j = 2 * (ab_idx % 2) + lj
                    U[global_i * 4 + global_j][col] += coeff * val
            
            # Construct V for this column
            # V has size 28 (4x7)
            # But we are only filling the first 4 columns of B (B1).
            # B1 is 4x4.
            for cd_idx, coeff in enumerate(outer_B[s]):
                if coeff == 0: continue
                # cd_idx: 0->B00, 1->B01, 2->B10, 3->B11
                # Block row: cd_idx // 2 (j block)
                # Block col: cd_idx % 2 (k block)
                for local_idx, val in enumerate(v_inner):
                    if val == 0: continue
                    li = local_idx // 2
                    lj = local_idx % 2
                    global_j = 2 * (cd_idx // 2) + li
                    global_k = 2 * (cd_idx % 2) + lj
                    # V index: j * p + k. p=7.
                    V[global_j * 7 + global_k][col] += coeff * val
            
            # Construct W for this column
            # W has size 28 (7x4)
            # We are filling the part corresponding to C1 (4x4).
            # C1 is 4x4.
            # outer_C[s] needs to be defined.
            
            # Let's define outer_C properly.
            # s=1 (M1): C00(+1), C11(+1)
            # s=2 (M2): C10(+1), C11(-1)
            # s=3 (M3): C01(+1), C11(+1)
            # s=4 (M4): C00(+1), C10(+1)
            # s=5 (M5): C00(-1), C01(+1)
            # s=6 (M6): C11(+1)
            # s=7 (M7): C00(+1)
            
            if s == 0:
                outer_C_s = [1, 0, 0, 1]
            elif s == 1:
                outer_C_s = [0, 0, 1, -1]
            elif s == 2:
                outer_C_s = [0, 1, 0, 1]
            elif s == 3:
                outer_C_s = [1, 0, 1, 0]
            elif s == 4:
                outer_C_s = [-1, 1, 0, 0]
            elif s == 5:
                outer_C_s = [0, 0, 0, 1]
            else:
                outer_C_s = [1, 0, 0, 0]
                
            for ef_idx, coeff in enumerate(outer_C_s):
                if coeff == 0: continue
                # ef_idx: 0->C00, 1->C01, 2->C10, 3->C11
                # Block row: ef_idx // 2 (k block)
                # Block col: ef_idx % 2 (i block)
                for local_idx, val in enumerate(w_inner):
                    if val == 0: continue
                    li = local_idx // 2
                    lj = local_idx % 2
                    global_k = 2 * (ef_idx // 2) + li
                    global_i = 2 * (ef_idx % 2) + lj
                    # W index: k * n + i. n=4.
                    W[global_k * 4 + global_i][col] += coeff * val
                    
            col += 1
            
    # Schoolbook for 4x4x3 part (B2)
    # B2 is columns 4,5,6 of B.
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j][col] = 1
                # V index: j * 7 + (4 + k)
                V[j * 7 + 4 + k][col] = 1
                # W index: (4 + k) * 4 + i
                W[(4 + k) * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
