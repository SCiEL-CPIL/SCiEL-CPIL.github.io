import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target Rank 40
    # Structure:
    # Block 1: A B1 (B1 is 3x3, cols 0-2 of B) -> Rank 17
    # Block 2: A B2 (B2 is 3x3, cols 3-5 of B) -> Rank 17
    # Block 3: A B3 (B3 is 3x1, col 6 of B) -> Rank 6
    # Total Rank 40
    
    rank = 40
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    start_time = time.time()
    deadline = start_time + 110

    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(len(u_vec)):
            if u_vec[i] != 0:
                U[i][col] += u_vec[i]
        for i in range(len(v_vec)):
            if v_vec[i] != 0:
                V[i][col] += v_vec[i]
        for i in range(len(w_vec)):
            if w_vec[i] != 0:
                W[i][col] += w_vec[i]
        col += 1
        if time.time() > deadline:
            raise TimeoutError()

    # Helper to add a Strassen 2x2x2 block
    # a_i_j: indices in global U for A block
    # b_j_k: indices in global V for B block
    # c_i_k: indices in global W for C block
    def add_strassen_222(a_idx, b_idx, c_idx):
        # a_idx: list of 4 indices [A00, A01, A10, A11] in U
        # b_idx: list of 4 indices [B00, B01, B10, B11] in V
        # c_idx: list of 4 indices [C00, C01, C10, C11] in W
        
        # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1] # C00, C11
        # Map local u/v/w to global
        gu = [0.0] * (n*m)
        gv = [0.0] * (m*p)
        gw = [0.0] * (p*n)
        for i in range(4):
            if u[i] != 0: gu[a_idx[i]] += u[i]
            if v[i] != 0: gv[b_idx[i]] += v[i]
            if w[i] != 0: gw[c_idx[i]] += w[i]
        add_term(gu, gv, gw)
        
        # P2 = A10(B00+B01) -> C10 += P2
        u = [0, 0, 1, 0]
        v = [1, 1, 0, 0]
        w = [0, 0, 1, 0] # C10
        gu = [0.0] * (n*m)
        gv = [0.0] * (m*p)
        gw = [0.0] * (p*n)
        for i in range(4):
            if u[i] != 0: gu[a_idx[i]] += u[i]
            if v[i] != 0: gv[b_idx[i]] += v[i]
            if w[i] != 0: gw[c_idx[i]] += w[i]
        add_term(gu, gv, gw)

        # P3 = A01(B10+B11) -> C01 += P3
        u = [0, 1, 0, 0]
        v = [0, 0, 1, 1]
        w = [0, 1, 0, 0] # C01
        gu = [0.0] * (n*m)
        gv = [0.0] * (m*p)
        gw = [0.0] * (p*n)
        for i in range(4):
            if u[i] != 0: gu[a_idx[i]] += u[i]
            if v[i] != 0: gv[b_idx[i]] += v[i]
            if w[i] != 0: gw[c_idx[i]] += w[i]
        add_term(gu, gv, gw)

        # P4 = A00(B00-B10) -> C10 += P4
        u = [1, 0, 0, 0]
        v = [1, 0, -1, 0]
        w = [0, 0, 1, 0] # C10
        gu = [0.0] * (n*m)
        gv = [0.0] * (m*p)
        gw = [0.0] * (p*n)
        for i in range(4):
            if u[i] != 0: gu[a_idx[i]] += u[i]
            if v[i] != 0: gv[b_idx[i]] += v[i]
            if w[i] != 0: gw[c_idx[i]] += w[i]
        add_term(gu, gv, gw)

        # P5 = A11(B01-B00) -> C01 += P5, C11 -= P5
        u = [0, 0, 0, 1]
        v = [0, 1, 0, -1] # B01 - B00? No, P5 = A11(B01-B00) is not standard.
        # Standard Strassen:
        # P5 = A11(B01 - B00) is incorrect.
        # P5 = A11(B01 - B00) -> C01 += P5, C11 -= P5?
        # Let's stick to:
        # P1 = (A11+A22)(B11+B22)
        # P2 = A21(B11+B12)
        # P3 = A12(B21+B22)
        # P4 = A11(B11-B21)
        # P5 = A22(B12-B11)
        # P6 = (A11+A12)B22
        # P7 = (A21-A11)B11
        # C11 = P1 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P6
        
        # My indices:
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # P5 = A11(B01-B00) -> u: [0,0,0,1], v: [0,1,0,-1]
        # C01 += P5, C11 -= P5
        # w: [0, 1, 0, -1]
        gu = [0.0] * (n*m)
        gv = [0.0] * (m*p)
        gw = [0.0] * (p*n)
        u = [0, 0, 0, 1]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, -1]
        for i in range(4):
            if u[i] != 0: gu[a_idx[i]] += u[i]
            if v[i] != 0: gv[b_idx[i]] += v[i]
            if w[i] != 0: gw[c_idx[i]] += w[i]
        add_term(gu, gv, gw)

        # P6 = (A00+A01)B11 -> C11 += P6
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [0, 0, 0, 1]
        gu = [0.0] * (n*m)
        gv = [0.0] * (m*p)
        gw = [0.0] * (p*n)
        for i in range(4):
            if u[i] != 0: gu[a_idx[i]] += u[i]
            if v[i] != 0: gv[b_idx[i]] += v[i]
            if w[i] != 0: gw[c_idx[i]] += w[i]
        add_term(gu, gv, gw)

        # P7 = (A10-A00)B00 -> C00 += P7
        u = [-1, 0, 1, 0]
        v = [1, 0, 0, 0]
        w = [1, 0, 0, 0]
        gu = [0.0] * (n*m)
        gv = [0.0] * (m*p)
        gw = [0.0] * (p*n)
        for i in range(4):
            if u[i] != 0: gu[a_idx[i]] += u[i]
            if v[i] != 0: gv[b_idx[i]] += v[i]
            if w[i] != 0: gw[c_idx[i]] += w[i]
        add_term(gu, gv, gw)

    def add_elm_221(a_idx, b_idx, c_idx):
        # A 2x2, B 2x1, C 2x1
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        # T1: A00 B00 -> C00
        gu = [0.0]*(n*m); gv = [0.0]*(m*p); gw = [0.0]*(p*n)
        gu[a_idx[0]] = 1; gv[b_idx[0]] = 1; gw[c_idx[0]] = 1
        add_term(gu, gv, gw)
        
        # T2: A01 B10 -> C00
        gu = [0.0]*(n*m); gv = [0.0]*(m*p); gw = [0.0]*(p*n)
        gu[a_idx[1]] = 1; gv[b_idx[2]] = 1; gw[c_idx[0]] = 1
        add_term(gu, gv, gw)
        
        # T3: A10 B00 -> C10
        gu = [0.0]*(n*m); gv = [0.0]*(m*p); gw = [0.0]*(p*n)
        gu[a_idx[2]] = 1; gv[b_idx[0]] = 1; gw[c_idx[2]] = 1
        add_term(gu, gv, gw)
        
        # T4: A11 B10 -> C10
        gu = [0.0]*(n*m); gv = [0.0]*(m*p); gw = [0.0]*(p*n)
        gu[a_idx[3]] = 1; gv[b_idx[2]] = 1; gw[c_idx[2]] = 1
        add_term(gu, gv, gw)

    def add_elm_213(a_idx, b_idx, c_idx):
        # A 2x1, B 1x3, C 2x3
        # C00 = A00 B00, C01 = A00 B01, C02 = A00 B02
        # C10 = A10 B00, C11 = A10 B01, C12 = A10 B02
        
        # T1: A00 B00 -> C00
        gu = [0.0]*(n*m); gv = [0.0]*(m*p); gw = [0.0]*(p*n)
        gu[a_idx[0]] = 1; gv[b_idx[0]] = 1; gw[c_idx[0]] = 1
        add_term(gu, gv, gw)
        
        # T2: A00 B01 -> C01
        gu = [0.0]*(n*m); gv = [0.0]*(m*p); gw = [0.0]*(p*n)
        gu[a_idx[0]] = 1; gv[b_idx[1]] = 1; gw[c_idx[1]] = 1
        add_term(gu, gv, gw)
        
        # T3: A00 B02 -> C02
        gu = [0.0]*(n*m); gv = [0.0]*(m*p); gw = [0.0]*(p*n)
        gu[a_idx[0]] = 1; gv[b_idx[2]] = 1; gw[c_idx[2]] = 1
        add_term(gu, gv, gw)
        
        # T4: A10 B00 -> C10
        gu = [0.0]*(n*m); gv = [0.0]*(m*p); gw = [0.0]*(p*n)
        gu[a_idx[1]] = 1; gv[b_idx[0]] = 1; gw[c_idx[3]] = 1
        add_term(gu, gv, gw)
        
        # T5: A10 B01 -> C11
        gu = [0.0]*(n*m); gv = [0.0]*(m*p); gw = [0.0]*(p*n)
        gu[a_idx[1]] = 1; gv[b_idx[1]] = 1; gw[c_idx[4]] = 1
        add_term(gu, gv, gw)
        
        # T6: A10 B02 -> C12
        gu = [0.0]*(n*m); gv = [0.0]*(m*p); gw = [0.0]*(p*n)
        gu[a_idx[1]] = 1; gv[b_idx[2]] = 1; gw[c_idx[5]] = 1
        add_term(gu, gv, gw)

    def add_elm_231(a_idx, b_idx, c_idx):
        # A 2x3, B 3x1, C 2x1
        # C00 = A00 B00 + A01 B10 + A02 B20
        # C10 = A10 B00 + A11 B10 + A12 B20
        
        for i in range(2):
            for j in range(3):
                gu = [0.0]*(n*m); gv = [0.0]*(m*p); gw = [0.0]*(p*n)
                gu[a_idx[i*3 + j]] = 1
                gv[b_idx[j]] = 1
                gw[c_idx[i]] = 1
                add_term(gu, gv, gw)

    # --- Block 1: A B1 ---
    # A indices: i in 0..1, j in 0..2
    # U index for A[i,j] is i*m + j = i*3 + j
    a_block1_22 = [0*3+0, 0*3+1, 1*3+0, 1*3+1] # A00, A01, A10, A11
    a_block1_21 = [0*3+2, 1*3+2] # A02, A12
    
    # B1 indices: j in 0..2, k in 0..2
    # V index for B[j,k] is j*p + k
    b_sub1_22 = [0*7+0, 0*7+1, 1*7+0, 1*7+1] # B00, B01, B10, B11
    b_sub1_21 = [0*7+2, 1*7+2] # B02, B12
    b_sub2_13 = [2*7+0, 2*7+1, 2*7+2] # B20, B21, B22
    
    # C1 indices: i in 0..1, k in 0..2
    # W index for C[i,k] is k*n + i (Trace convention: W[k*n+i, r])
    c_block1_22 = [0*2+0, 0*2+1, 1*2+0, 1*2+1] # C00, C01, C10, C11
    c_block1_21 = [2*2+0, 2*2+1] # C02, C12
    
    # M_223 for A_22 B_23
    # Split B_23 into B_22 (cols 0,1) and B_21 (col 2)
    b_22_22 = b_sub1_22
    b_22_21 = b_sub1_21
    c_22_22 = c_block1_22
    c_22_21 = c_block1_21
    
    # M_222 (Strassen)
    add_strassen_222(a_block1_22, b_22_22, c_22_22)
    # M_221 (Elementary)
    add_elm_221(a_block1_22, b_22_21, c_22_21)
    # M_213 (Elementary)
    add_elm_213(a_block1_21, b_sub2_13, c_block1_22) # Note: Output C is 2x3, so c_idx should be 6 elements?
    # Wait, add_elm_213 expects c_idx of length 6.
    # c_block1_22 is only 4 elements.
    # My add_elm_213 writes to c_idx[0..5].
    # I need to provide 6 c indices.
    # C1 is 2x3. Indices:
    # C00 (0), C01 (1), C02 (2), C10 (3), C11 (4), C12 (5)
    # W indices:
    # C00: k=0, i=0 -> 0*2+0=0
    # C01: k=1, i=0 -> 1*2+0=2
    # C02: k=2, i=0 -> 2*2+0=4
    # C10: k=0, i=1 -> 0*2+1=1
    # C11: k=1, i=1 -> 1*2+1=3
    # C12: k=2, i=1 -> 2*2+1=5
    
    # Let's restart the index mapping for W carefully.
    # W index = k * n + i.
    
    # Correct W indices for 2x3 block:
    # C00: 0
    # C01: 2
    # C02: 4
    # C10: 1
    # C11: 3
    # C12: 5
    
    # My previous c_block1_22 was [0, 2, 1, 3] for 2x2?
    # C00: 0
    # C01: 2
    # C10: 1
    # C11: 3
    # Yes.
    
    # So for add_elm_213, I need to pass [0, 2, 4, 1, 3, 5] as c_idx.
    
    # I need to re-run the logic with correct indices.

    # Reset U, V, W
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    
    # Re-define helper with correct W index logic
    # Actually, I will just hardcode the W indices in the calls.

    # Block 1: A B1
    # Strassen 2x2x2 on A[:,0:2], B[0:2,0:2]
    # W indices for C[0:2,0:2]:
    # C00:0, C01:2, C10:1, C11:3
    w_22 = [0, 2, 1, 3]
    
    # M_221 on A[:,0:2], B[0:2,2]
    # W indices for C[0:2,2]:
    # C02:4, C12:5
    w_21 = [4, 5]
    
    # M_213 on A[:,2], B[2,0:3]
    # W indices for C[0:2,0:3]:
    # C00:0, C01:2, C02:4, C10:1, C11:3, C12:5
    w_23 = [0, 2, 4, 1, 3, 5]
    
    a_22 = [0, 1, 3, 4] # A00, A01, A10, A11
    a_21 = [2, 5]       # A02, A12
    
    b_22 = [0, 1, 7, 8] # B00, B01, B10, B11
    b_21 = [2, 9]       # B02, B12
    b_13 = [14, 15, 16] # B20, B21, B22
    
    add_strassen_222(a_22, b_22, w_22)
    add_elm_221(a_22, b_21, w_21)
    add_elm_213(a_21, b_13, w_23)
    
    # Block 2: A B2 (Cols 3-5 of B)
    # B2 rows 0-1, cols 3-4: B[0:2, 3:5]
    # B2 rows 0-1, col 5: B[0:2, 5]
    # B2 row 2, cols 3-5: B[2, 3:6]
    
    # W indices for C[0:2, 3:6]:
    # C03: 3*2+0=6, C04: 4*2+0=8, C05: 5*2+0=10
    # C13: 3*2+1=7, C14: 4*2+1=9, C15: 5*2+1=11
    w_22_2 = [6, 8, 7, 9]
    w_21_2 = [10, 11]
    w_23_2 = [6, 8, 10, 7, 9, 11]
    
    b_22_2 = [3, 4, 10, 11] # B03, B04, B13, B14
    b_21_2 = [5, 12]        # B05, B15
    b_13_2 = [17, 18, 19]   # B23, B24, B25
    
    add_strassen_222(a_22, b_22_2, w_22_2)
    add_elm_221(a_22, b_21_2, w_21_2)
    add_elm_213(a_21, b_13_2, w_23_2)
    
    # Block 3: A B3 (Col 6 of B)
    # A 2x3, B 3x1, C 2x1
    # W indices for C[0:2, 6]:
    # C06: 6*2+0=12
    # C16: 6*2+1=13
    w_31 = [12, 13]
    
    a_23 = [0, 1, 2, 3, 4, 5]
    b_31 = [6, 13, 20] # B06, B16, B26
    
    add_elm_231(a_23, b_31, w_31)
    
    # Verify rank used
    if col != rank:
        print(f"Warning: Rank used {col} vs expected {rank}")
        
    return {"U": U, "V": V, "W": W}
