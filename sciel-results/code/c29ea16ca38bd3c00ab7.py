import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 77
    # We use recursive Strassen.
    # Level 1: 4x4 x 4x6 using 7 products of 2x2 x 2x3 matrices.
    # Level 2: 2x2 x 2x3 using 7 products of 1x1 x 1x1 (Strassen for 2x2x2 part) 
    #          + 4 products of 1x1 x 1x1 (Schoolbook for 2x2x1 part).
    # Total rank 7 * (7+4) = 77.
    
    # --- Step 1: Define 2x2x2 Strassen (Rank 7) ---
    # A is 2x2, B is 2x2, C is 2x2.
    # Indices: A: 0..3, B: 0..3, C: 0..3.
    # A = [[a0, a1], [a2, a3]]
    # B = [[b0, b1], [b2, b3]]
    # C = [[c0, c1], [c2, c3]]
    
    U_222 = [[0]*7 for _ in range(4)]
    V_222 = [[0]*7 for _ in range(4)]
    W_222 = [[0]*7 for _ in range(4)]
    
    # Strassen formulas
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)b0
    # M3 = a0(b1-b3)
    # M4 = a3(b2-b0)
    # M5 = (a0+a1)b3
    # M6 = (a2+a3)(b1+b3)
    # M7 = (a0-a2)(b0+b1)
    
    # U (A factors)
    # r=0: a0+a3 -> [1, 0, 0, 1]
    U_222[0][0] = 1; U_222[3][0] = 1
    # r=1: a2+a3 -> [0, 0, 1, 1]
    U_222[2][1] = 1; U_222[3][1] = 1
    # r=2: a0 -> [1, 0, 0, 0]
    U_222[0][2] = 1
    # r=3: a3 -> [0, 0, 0, 1]
    U_222[3][3] = 1
    # r=4: a0+a1 -> [1, 1, 0, 0]
    U_222[0][4] = 1; U_222[1][4] = 1
    # r=5: a2+a3 -> [0, 0, 1, 1]
    U_222[2][5] = 1; U_222[3][5] = 1
    # r=6: a0-a2 -> [1, 0, -1, 0]
    U_222[0][6] = 1; U_222[2][6] = -1
    
    # V (B factors)
    # r=0: b0+b3 -> [1, 0, 0, 1]
    V_222[0][0] = 1; V_222[3][0] = 1
    # r=1: b0 -> [1, 0, 0, 0]
    V_222[0][1] = 1
    # r=2: b1-b3 -> [0, 1, 0, -1]
    V_222[1][2] = 1; V_222[3][2] = -1
    # r=3: b2-b0 -> [ -1, 0, 1, 0]
    V_222[0][3] = -1; V_222[2][3] = 1
    # r=4: b3 -> [0, 0, 0, 1]
    V_222[3][4] = 1
    # r=5: b1+b3 -> [0, 1, 0, 1]
    V_222[1][5] = 1; V_222[3][5] = 1
    # r=6: b0+b1 -> [1, 1, 0, 0]
    V_222[0][6] = 1; V_222[1][6] = 1
    
    # W (C factors)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # r=0: M1 -> C00 (+1), C11 (+1)
    W_222[0][0] = 1; W_222[3][0] = 1
    # r=1: M2 -> C10 (+1), C11 (-1)
    W_222[2][1] = 1; W_222[3][1] = -1
    # r=2: M3 -> C01 (+1), C11 (+1)
    W_222[1][2] = 1; W_222[3][2] = 1
    # r=3: M4 -> C00 (-1), C10 (+1)
    W_222[0][3] = -1; W_222[2][3] = 1
    # r=4: M5 -> C00 (-1), C01 (+1)
    W_222[0][4] = -1; W_222[1][4] = 1
    # r=5: M6 -> C11 (+1)
    W_222[3][5] = 1
    # r=6: M7 -> C00 (-1)
    W_222[0][6] = -1

    # --- Step 2: Define 2x2x3 Decomposition (Rank 11) ---
    # A is 2x2 (4 entries).
    # B is 2x3 (6 entries).
    # C is 2x3 (6 entries).
    
    U_223 = [[0]*11 for _ in range(4)]
    V_223 = [[0]*11 for _ in range(6)]
    W_223 = [[0]*11 for _ in range(6)]
    
    # Part 1: Strassen for cols 0,1 (Rank 7)
    # Map V_222 (indices j*2+k) to V_223 (indices j*3+k)
    # j=0,k=0 -> 0. j=0,k=1 -> 1.
    # j=1,k=0 -> 3. j=1,k=1 -> 4.
    
    for r in range(7):
        # U is same as U_222
        for i in range(4):
            U_223[i][r] = U_222[i][r]
        
        # V mapping
        # V_222 index: j*2+k
        for j in range(2):
            for k in range(2):
                idx_222 = j*2 + k
                idx_223 = j*3 + k
                V_223[idx_223][r] = V_222[idx_222][r]
                
        # W mapping
        # W_222 index: k*2+i (Wait, W index in 222 was k*2+i? No, C is 2x2. 
        # In my U/V/W definition above, I used row-major for A and B?
        # Let's check indices.
        # A: i*m+j. 2x2 -> i*2+j.
        # B: j*p+k. 2x2 -> j*2+k.
        # C: k*n+i. 2x2 -> k*2+i.
        # My W_222 above used indices 0..3.
        # W_222[0] is C00 (k=0,i=0). W_222[1] is C01 (k=0,i=1).
        # W_222[2] is C10 (k=1,i=0). W_222[3] is C11 (k=1,i=1).
        # So W_222 index is k*2+i.
        
        for k in range(2):
            for i in range(2):
                idx_222 = k*2 + i
                idx_223 = k*3 + i
                W_223[idx_223][r] = W_222[idx_222][r]

    # Part 2: Schoolbook for col 2 (Rank 4)
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # A indices: 0(A00), 1(A01), 2(A10), 3(A11).
    # B indices: j*3+k.
    # B02: j=0,k=2 -> 2.
    # B12: j=1,k=2 -> 5.
    # C indices: k*3+i.
    # C02: k=0,i=2 -> 2.
    # C12: k=1,i=2 -> 5.
    
    r_offset = 7
    # r=7: A00 * B02 -> C02
    U_223[0][r_offset] = 1
    V_223[2][r_offset] = 1
    W_223[2][r_offset] = 1
    
    # r=8: A01 * B12 -> C02
    U_223[1][r_offset+1] = 1
    V_223[5][r_offset+1] = 1
    W_223[2][r_offset+1] = 1
    
    # r=9: A10 * B02 -> C12
    U_223[2][r_offset+2] = 1
    V_223[2][r_offset+2] = 1
    W_223[5][r_offset+2] = 1
    
    # r=10: A11 * B12 -> C12
    U_223[3][r_offset+3] = 1
    V_223[5][r_offset+3] = 1
    W_223[5][r_offset+3] = 1

    # --- Step 3: Construct 4x4x6 Decomposition (Rank 77) ---
    # A is 4x4. B is 4x6.
    # Strassen on blocks.
    # A blocks: A00, A01, A10, A11 (each 2x2).
    # B blocks: B00, B01, B10, B11 (each 2x3).
    
    U_446 = [[0]*77 for _ in range(16)]
    V_446 = [[0]*77 for _ in range(24)]
    W_446 = [[0]*77 for _ in range(24)]
    
    # Strassen 4x4x4 logic (adapted for 2x3 blocks)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10+A11)(B01+B11)
    # M7 = (A00-A10)(B00+B01)
    
    # We will fill U, V, W for each of the 7 Strassen steps.
    # Each step uses the 223 decomposition (Rank 11).
    
    # Helper to map 2x2 block indices to 4x4 indices
    # A: i*4+j. Block (bi, bj) is 2x2.
    # Local index (li, lj) in block -> Global (bi*2+li, bj*2+lj) -> (bi*2+li)*4 + (bj*2+lj)
    
    # Helper to map 2x3 block indices to 4x6 indices
    # B: j*6+k. Block (bj, bk) is 2x3.
    # Local index (lj, lk) in block -> Global (bj*2+lj, bk*3+lk) -> (bj*2+lj)*6 + (bk*3+lk)
    
    # Helper to map 2x3 block indices to 4x6 output indices
    # C: k*4+i. Block (ck, ci) is 2x3.
    # Local index (lk, li) in block -> Global (ck*2+lk, ci*2+li) -> (ck*2+lk)*4 + (ci*2+li)
    
    # Strassen Step 1: M1 = (A00+A11)(B00+B11)
    # A part: A00 + A11.
    # B part: B00 + B11.
    
    # We need to add the 223 decomposition for each Strassen term.
    # The 223 decomposition computes C_block = A_block * B_block.
    # Here A_block is a linear combo, B_block is a linear combo.
    
    # Let's define the linear combinations for A and B for each Strassen step.
    # A_comb: dict of (block_i, block_j) -> coeff
    # B_comb: dict of (block_j, block_k) -> coeff
    
    steps = []
    # M1
    steps.append((
        {(0,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,1): 1}
    ))
    # M2
    steps.append((
        {(1,0): 1, (1,1): 1}, 
        {(0,0): 1}
    ))
    # M3
    steps.append((
        {(0,0): 1}, 
        {(0,1): 1, (1,1): -1}
    ))
    # M4
    steps.append((
        {(1,1): 1}, 
        {(1,0): 1, (0,0): -1}
    ))
    # M5
    steps.append((
        {(0,0): 1, (0,1): 1}, 
        {(1,1): 1}
    ))
    # M6
    steps.append((
        {(1,0): 1, (1,1): 1}, 
        {(0,1): 1, (1,1): 1}
    ))
    # M7
    steps.append((
        {(0,0): 1, (1,0): -1}, 
        {(0,0): 1, (0,1): 1}
    ))
    
    for s_idx, (a_comb, b_comb) in enumerate(steps):
        base_r = s_idx * 11
        for r_local in range(11):
            r_global = base_r + r_local
            
            # Fill U
            for (bi, bj), coeff_a in a_comb.items():
                # Local A index in 223: li*2 + lj
                for li in range(2):
                    for lj in range(2):
                        u_val = U_223[li*2 + lj][r_local]
                        if u_val != 0:
                            gi = bi*2 + li
                            gj = bj*2 + lj
                            g_idx = gi*4 + gj
                            U_446[g_idx][r_global] += coeff_a * u_val
            
            # Fill V
            for (bj, bk), coeff_b in b_comb.items():
                # Local B index in 223: lj*3 + lk
                for lj in range(2):
                    for lk in range(3):
                        v_val = V_223[lj*3 + lk][r_local]
                        if v_val != 0:
                            gj = bj*2 + lj
                            gk = bk*3 + lk
                            g_idx = gj*6 + gk
                            V_446[g_idx][r_global] += coeff_b * v_val
            
            # Fill W
            # Output C is 4x6.
            # The Strassen step produces a 2x3 matrix M_s.
            # M_s is added to C blocks.
            # Which C blocks?
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # We need to map the local 223 W indices to global 446 W indices.
            # Local W index in 223: lk*3 + li (Wait, C is 2x3. k is row, i is col?
            # In 223, C is 2x3. Indices k*3+i?
            # My W_223 index was k*3+i.
            # Global C is 4x6. Indices k*4+i.
            
            # The Strassen step s contributes to specific C blocks.
            # Let's define the contribution coeffs for C blocks.
            # c_comb: dict of (ck, ci) -> coeff
            
            if s_idx == 0: # M1
                c_comb = {(0,0): 1, (1,1): 1}
            elif s_idx == 1: # M2
                c_comb = {(1,0): 1, (1,1): -1}
            elif s_idx == 2: # M3
                c_comb = {(0,1): 1, (1,1): 1}
            elif s_idx == 3: # M4
                c_comb = {(0,0): -1, (1,0): 1}
            elif s_idx == 4: # M5
                c_comb = {(0,0): -1, (0,1): 1}
            elif s_idx == 5: # M6
                c_comb = {(1,1): 1}
            elif s_idx == 6: # M7
                c_comb = {(0,0): -1}
            
            for (ck, ci), coeff_c in c_comb.items():
                # Local C index in 223: lk*3 + li
                for lk in range(2):
                    for li in range(3):
                        w_val = W_223[lk*3 + li][r_local]
                        if w_val != 0:
                            gk = ck*2 + lk
                            gi = ci*2 + li
                            g_idx = gk*4 + gi
                            W_446[g_idx][r_global] += coeff_c * w_val

    return {"U": U_446, "V": V_446, "W": W_446}
