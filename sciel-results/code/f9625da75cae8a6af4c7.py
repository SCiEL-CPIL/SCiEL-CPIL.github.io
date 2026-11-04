import numpy as np

def construct():
    n, m, p = 3, 4, 5
    
    # 1. Generate Strassen 2x2 factors
    # U2, V2, W2 shapes (4, 7)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    # M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    # M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    # M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    # M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    # M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1

    # 2. Construct Strassen 4x4 factors
    # Rank 49
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Strassen block indices for 4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We iterate over the 7 Strassen steps (s) and 7 internal factors (t)
    # s=0..6, t=0..6
    # Column c = s*7 + t
    
    # Define which A-block, B-block, C-block are involved in each Strassen step s
    # A-blocks: 00, 01, 10, 11
    # B-blocks: 00, 01, 10, 11
    # C-blocks: 00, 01, 10, 11
    
    # For each step s, we have a linear combination of A-blocks and B-blocks.
    # The internal factor t picks specific elements from these blocks.
    
    # Let's map the 2x2 indices (0..3) to 4x4 indices (0..15)
    # Block 00: rows 0-1, cols 0-1. Indices 0,1,4,5
    # Block 01: rows 0-1, cols 2-3. Indices 2,3,6,7
    # Block 10: rows 2-3, cols 0-1. Indices 8,9,12,13
    # Block 11: rows 2-3, cols 2-3. Indices 10,11,14,15
    
    # Helper to get 4x4 index from block (br, bc) and 2x2 index (r, c)
    def idx4(br, bc, r, c):
        return (br*2 + r)*4 + (bc*2 + c)
        
    # Strassen Steps Definition
    # Each step s has:
    # - A terms: list of (block, coeff)
    # - B terms: list of (block, coeff)
    # - C terms: list of (block, coeff)
    
    steps = [
        # M1
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        # M2
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(1, 1), (3, -1)]},
        # M3
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
        # M4
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (1, 1)]},
        # M5
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        # M6
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        # M7
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
    ]
    
    for s in range(7):
        step = steps[s]
        A_terms = step['A']
        B_terms = step['B']
        C_terms = step['C']
        
        for t in range(7):
            c = s*7 + t
            
            # Determine which 2x2 indices are active in U2, V2, W2 for this t
            # U2[i, t] != 0
            for i2 in range(4):
                if U2[i2][t] != 0:
                    u_val = U2[i2][t]
                    r2 = i2 // 2
                    c2 = i2 % 2
                    for (ab, a_coeff) in A_terms:
                        # ab is block index 0..3
                        abr = ab // 2
                        abc = ab % 2
                        idx = idx4(abr, abc, r2, c2)
                        U4[idx][c] += u_val * a_coeff
                        
            for j2 in range(4):
                if V2[j2][t] != 0:
                    v_val = V2[j2][t]
                    r2 = j2 // 2
                    c2 = j2 % 2
                    for (bb, b_coeff) in B_terms:
                        bbr = bb // 2
                        bbc = bb % 2
                        idx = idx4(bbr, bbc, r2, c2)
                        V4[idx][c] += v_val * b_coeff
                        
            for k2 in range(4):
                if W2[k2][t] != 0:
                    w_val = W2[k2][t]
                    r2 = k2 // 2
                    c2 = k2 % 2
                    for (cb, c_coeff) in C_terms:
                        cbr = cb // 2
                        cbc = cb % 2
                        idx = idx4(cbr, cbc, r2, c2)
                        W4[idx][c] += w_val * c_coeff

    # 3. Construct Small 3x4x1 factors
    # Rank 1
    U_small = [[1]*1 for _ in range(12)] # i*4+j, all 1
    V_small = [[0]*1 for _ in range(20)]
    for j in range(4):
        V_small[j*5 + 4][0] = 1 # j*5 + k, k=4
    W_small = [[0]*1 for _ in range(15)]
    for i in range(3):
        W_small[4*3 + i][0] = 1 # k*3+i, k=4

    # 4. Assemble Final Factors
    rank = 50
    U = [[0]*rank for _ in range(12)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(15)]
    
    # Copy Strassen part (cols 0-48)
    for c in range(49):
        for i in range(12):
            U[i][c] = U4[i][c]
        for j in range(4):
            for k in range(4):
                V[j*5 + k][c] = V4[j*4 + k][c]
        for k in range(4):
            for i in range(3):
                W[k*3 + i][c] = W4[k*4 + i][c]
                
    # Copy Small part (col 49)
    for i in range(12):
        U[i][49] = U_small[i][0]
    for j in range(4):
        V[j*5 + 4][49] = V_small[j*5 + 4][0]
    for i in range(3):
        W[4*3 + i][49] = W_small[4*3 + i][0]

    return {"U": U, "V": V, "W": W}
