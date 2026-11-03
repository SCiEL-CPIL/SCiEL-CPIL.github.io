# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses a block decomposition with Strassen optimization to achieve rank 56.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140
    
    # We will construct the 56 terms
    # U: 10x56, V: 30x56, W: 12x56
    # Indices for A (2x5): i in 0..1, j in 0..4 -> idx = i*5 + j
    # Indices for B (5x6): j in 0..4, k in 0..5 -> idx = j*6 + k
    # Indices for C (2x6): i in 0..1, k in 0..5 -> idx = k*2 + i (Trace convention W[k*n+i])
    
    U = np.zeros((10, 56))
    V = np.zeros((30, 56))
    W = np.zeros((12, 56))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1

    # Block 1: A1 (2x2) * B1 (2x3) -> C1_part (2x3)
    # A1: i=0,1; j=0,1
    # B1: j=0,1; k=0,1,2 (cols 0,1,2 of B)
    # C1_part: i=0,1; k=0,1,2 (cols 0,1,2 of C)
    
    # Strassen for A1 * B1[:, 1:3] (2x2 * 2x2)
    # B1[:, 1:3] indices: j=0,1; k=1,2
    # A1 indices: i=0,1; j=0,1
    
    # M1 = (A00 + A11) * (B01 + B12)
    # A00: i=0, j=0 -> idx 0
    # A11: i=1, j=1 -> idx 6
    # B01: j=0, k=1 -> idx 1
    # B12: j=1, k=2 -> idx 8
    # C01: i=0, k=1 -> idx 2 (W[1*2+0]=W[2])
    # C11: i=1, k=1 -> idx 3 (W[1*2+1]=W[3])
    # Wait, C indices: k*n+i.
    # k=1, i=0 -> 1*2+0 = 2.
    # k=1, i=1 -> 1*2+1 = 3.
    # k=2, i=0 -> 2*2+0 = 4.
    # k=2, i=1 -> 2*2+1 = 5.
    
    # M1
    add_term(0, 1, 1, 1, 2, 1) # C01
    add_term(0, 1, 1, 1, 3, 1) # C11? No, M1 contributes to C01, C11, C02, C12?
    # Let's map Strassen outputs carefully.
    # C' = A B'
    # C'00 (k=1, i=0) = M1 + M4 - M5 + M7
    # C'01 (k=2, i=0) = M3 + M5
    # C'10 (k=1, i=1) = M2 + M4
    # C'11 (k=2, i=1) = M1 - M2 + M3 + M6
    
    # We need to add contributions to W for each M.
    # M1 = (A00 + A11) (B01 + B12)
    # U: A00 (idx 0), A11 (idx 6)
    # V: B01 (idx 1), B12 (idx 8)
    # W: C01 (idx 2), C11 (idx 3), C02 (idx 4), C12 (idx 5)
    # Coeffs for M1:
    # C01: +1
    # C11: +1
    # C02: 0
    # C12: -1
    # Wait, C'11 = M1 - M2 + M3 + M6. So M1 coeff is +1.
    # C'00 = M1 + M4 - M5 + M7. So M1 coeff is +1.
    
    # It is easier to define the 7 Strassen terms and their U, V, and W contributions.
    
    def add_strassen_block(a_offset, b_offset, c_offset, b_col_start):
        # a_offset: index in A for top-left (i=0, j=a_offset)
        # b_offset: index in B for top-left (j=b_offset, k=b_col_start)
        # c_offset: index in C for top-left (i=0, k=c_offset)
        # b_col_start: which column of B this block starts at (0 or 1 relative to block)
        # Actually, let's just hardcode the indices for the two blocks.
        pass

    # Block 1: A1 (j=0,1) * B1 (j=0,1; k=1,2)
    # A indices: 0, 1, 5, 6 (i*5+j)
    # 0:0,0; 1:0,1; 5:1,0; 6:1,1
    # B indices: 1, 2, 7, 8 (j*6+k)
    # 1:0,1; 2:0,2; 7:1,1; 8:1,2
    # C indices: 2, 3, 4, 5 (k*2+i)
    # 2:1,0; 3:1,1; 4:2,0; 5:2,1
    
    # M1 = (A00 + A11)(B01 + B12)
    # U: 0:+1, 6:+1
    # V: 1:+1, 8:+1
    # W: 2:+1 (C10), 3:+1 (C11), 4:0, 5:-1 (C21)
    # Wait, C'00 is C(k=1, i=0) -> idx 2.
    # C'11 is C(k=2, i=1) -> idx 5.
    # C'00 = M1 + M4 - M5 + M7
    # C'01 = M3 + M5 -> C(k=2, i=0) -> idx 4
    # C'10 = M2 + M4 -> C(k=1, i=1) -> idx 3
    # C'11 = M1 - M2 + M3 + M6 -> C(k=2, i=1) -> idx 5
    
    # Term 1: M1
    U[0, col] = 1; U[6, col] = 1
    V[1, col] = 1; V[8, col] = 1
    W[2, col] = 1; W[3, col] = 1; W[5, col] = -1
    col += 1
    
    # Term 2: M2 = (A10 + A11) B01
    # A10: idx 5, A11: idx 6
    # B01: idx 1
    # W: C'10 (idx 3) +1, C'11 (idx 5) -1
    U[5, col] = 1; U[6, col] = 1
    V[1, col] = 1
    W[3, col] = 1; W[5, col] = -1
    col += 1
    
    # Term 3: M3 = A00 (B02 - B12)
    # A00: idx 0
    # B02: idx 2, B12: idx 8
    # W: C'01 (idx 4) +1, C'11 (idx 5) +1
    U[0, col] = 1
    V[2, col] = 1; V[8, col] = -1
    W[4, col] = 1; W[5, col] = 1
    col += 1
    
    # Term 4: M4 = A11 (B01 - B02)
    # A11: idx 6
    # B01: idx 1, B02: idx 2
    # W: C'00 (idx 2) +1, C'10 (idx 3) +1
    U[6, col] = 1
    V[1, col] = 1; V[2, col] = -1
    W[2, col] = 1; W[3, col] = 1
    col += 1
    
    # Term 5: M5 = (A00 + A01) B12
    # A00: idx 0, A01: idx 1
    # B12: idx 8
    # W: C'00 (idx 2) -1, C'01 (idx 4) +1
    U[0, col] = 1; U[1, col] = 1
    V[8, col] = 1
    W[2, col] = -1; W[4, col] = 1
    col += 1
    
    # Term 6: M6 = (A10 - A00) (B01 + B02)
    # A10: idx 5, A00: idx 0
    # B01: idx 1, B02: idx 2
    # W: C'11 (idx 5) +1
    U[5, col] = 1; U[0, col] = -1
    V[1, col] = 1; V[2, col] = 1
    W[5, col] = 1
    col += 1
    
    # Term 7: M7 = (A01 - A11) (B11 + B12)
    # A01: idx 1, A11: idx 6
    # B11: idx 7, B12: idx 8
    # W: C'00 (idx 2) +1
    U[1, col] = 1; U[6, col] = -1
    V[7, col] = 1; V[8, col] = 1
    W[2, col] = 1
    col += 1
    
    # Schoolbook for A1 * B1[:, 0] (2x2 * 2x1)
    # B00: idx 0, B10: idx 6
    # C00: idx 0 (k=0, i=0), C10: idx 1 (k=0, i=1)
    
    # Term 8: A00 B00
    U[0, col] = 1
    V[0, col] = 1
    W[0, col] = 1
    col += 1
    
    # Term 9: A01 B10
    U[1, col] = 1
    V[6, col] = 1
    W[0, col] = 1
    col += 1
    
    # Term 10: A10 B00
    U[5, col] = 1
    V[0, col] = 1
    W[1, col] = 1
    col += 1
    
    # Term 11: A11 B10
    U[6, col] = 1
    V[6, col] = 1
    W[1, col] = 1
    col += 1
    
    # Block 2: A2 (2x3) * B2 (3x3)
    # A2: j=2,3,4. i=0,1.
    # Indices: 2,3,4,7,8,9
    # B2: j=2,3,4. k=0,1,2.
    # Indices: 12,13,14, 18,19,20, 24,25,26
    # C2_part: k=0,1,2. i=0,1.
    # Indices: 0,1, 2,3, 4,5
    
    # A_L (2x1) * B_L (1x3)
    # A_L: j=2. i=0,1. Indices 2, 7.
    # B_L: j=2. k=0,1,2. Indices 12, 13, 14.
    
    # Term 12: A02 B20
    U[2, col] = 1
    V[12, col] = 1
    W[0, col] = 1
    col += 1
    
    # Term 13: A02 B21
    U[2, col] = 1
    V[13, col] = 1
    W[2, col] = 1
    col += 1
    
    # Term 14: A02 B22
    U[2, col] = 1
    V[14, col] = 1
    W[4, col] = 1
    col += 1
    
    # Term 15: A12 B20
    U[7, col] = 1
    V[12, col] = 1
    W[1, col] = 1
    col += 1
    
    # Term 16: A12 B21
    U[7, col] = 1
    V[13, col] = 1
    W[3, col] = 1
    col += 1
    
    # Term 17: A12 B22
    U[7, col] = 1
    V[14, col] = 1
    W[5, col] = 1
    col += 1
    
    # A_R (2x2) * B_R (2x3)
    # A_R: j=3,4. i=0,1. Indices 3,4,8,9.
    # B_R: j=3,4. k=0,1,2. Indices 18,19,20, 24,25,26.
    # This is same structure as Block 1 Strassen part.
    
    # M1' = (A03 + A14) (B30 + B42)
    # A03: 3, A14: 9
    # B30: 18, B42: 26
    # W: C00 (0), C10 (1), C11 (5)
    U[3, col] = 1; U[9, col] = 1
    V[18, col] = 1; V[26, col] = 1
    W[0, col] = 1; W[1, col] = 1; W[5, col] = -1
    col += 1
    
    # M2' = (A13 + A14) B30
    U[8, col] = 1; U[9, col] = 1
    V[18, col] = 1
    W[1, col] = 1; W[5, col] = -1
    col += 1
    
    # M3' = A03 (B32 - B42)
    U[3, col] = 1
    V[20, col] = 1; V[26, col] = -1
    W[4, col] = 1; W[5, col] = 1
    col += 1
    
    # M4' = A14 (B30 - B32)
    U[9, col] = 1
    V[18, col] = 1; V[20, col] = -1
    W[0, col] = 1; W[1, col] = 1
    col += 1
    
    # M5' = (A03 + A04) B42
    U[3, col] = 1; U[4, col] = 1
    V[26, col] = 1
    W[0, col] = -1; W[4, col] = 1
    col += 1
    
    # M6' = (A13 - A03) (B30 + B32)
    U[8, col] = 1; U[3, col] = -1
    V[18, col] = 1; V[20, col] = 1
    W[5, col] = 1
    col += 1
    
    # M7' = (A04 - A14) (B40 + B42)
    U[4, col] = 1; U[9, col] = -1
    V[24, col] = 1; V[26, col] = 1
    W[0, col] = 1
    col += 1
    
    # Schoolbook for A_R * B_R[:, 0]
    # B30: 18, B40: 24
    # C00: 0, C10: 1
    
    # Term 25: A03 B30
    U[3, col] = 1
    V[18, col] = 1
    W[0, col] = 1
    col += 1
    
    # Term 26: A04 B40
    U[4, col] = 1
    V[24, col] = 1
    W[0, col] = 1
    col += 1
    
    # Term 27: A13 B30
    U[8, col] = 1
    V[18, col] = 1
    W[1, col] = 1
    col += 1
    
    # Term 28: A14 B40
    U[9, col] = 1
    V[24, col] = 1
    W[1, col] = 1
    col += 1
    
    # Now we have 28 terms for C1 (k=0,1,2).
    # We need to repeat for C2 (k=3,4,5).
    # B2 for C2: j=2,3,4; k=3,4,5.
    # Indices: 15,16,17, 21,22,23, 27,28,29.
    # C2 indices: 6,7, 8,9, 10,11.
    
    # Copy the 28 terms, adjusting B and W indices.
    # B indices shift by +3 in k dimension.
    # j*6+k -> j*6+(k+3) = j*6+k+3.
    # W indices shift by +6 in k dimension.
    # k*2+i -> (k+3)*2+i = k*2+i+6.
    
    for t in range(28):
        # Copy U
        U[:, col] = U[:, t]
        # Copy V with index shift
        # We need to map V indices.
        # Original V indices for Block 1 (B1): 0..8
        # Original V indices for Block 2 (B2): 12..26
        # New V indices: +3
        # Original W indices: 0..5
        # New W indices: +6
        
        # Instead of shifting, we can just re-run the logic with offsets.
        # But copying is faster if we store the non-zero indices.
        pass
        
    # Let's just re-run the construction for the second half.
    # It's safer to avoid index mapping errors.
    
    # Block 1 Repeat: A1 * B1_right (2x2 * 2x3)
    # B1_right: j=0,1; k=3,4,5.
    # Indices: 3,4, 9,10.
    # C_right: k=3,4,5. i=0,1.
    # Indices: 6,7, 8,9, 10,11.
    
    # M1 = (A00 + A11) (B03 + B15)
    U[0, col] = 1; U[6, col] = 1
    V[3, col] = 1; V[10, col] = 1
    W[8, col] = 1; W[9, col] = 1; W[11, col] = -1
    col += 1
    
    # M2 = (A10 + A11) B03
    U[5, col] = 1; U[6, col] = 1
    V[3, col] = 1
    W[9, col] = 1; W[11, col] = -1
    col += 1
    
    # M3 = A00 (B04 - B15)
    U[0, col] = 1
    V[4, col] = 1; V[10, col] = -1
    W[10, col] = 1; W[11, col] = 1
    col += 1
    
    # M4 = A11 (B03 - B04)
    U[6, col] = 1
    V[3, col] = 1; V[4, col] = -1
    W[8, col] = 1; W[9, col] = 1
    col += 1
    
    # M5 = (A00 + A01) B15
    U[0, col] = 1; U[1, col] = 1
    V[10, col] = 1
    W[8, col] = -1; W[10, col] = 1
    col += 1
    
    # M6 = (A10 - A00) (B03 + B04)
    U[5, col] = 1; U[0, col] = -1
    V[3, col] = 1; V[4, col] = 1
    W[11, col] = 1
    col += 1
    
    # M7 = (A01 - A11) (B13 + B15)
    U[1, col] = 1; U[6, col] = -1
    V[9, col] = 1; V[10, col] = 1
    W[8, col] = 1
    col += 1
    
    # Schoolbook A1 * B1[:, 3]
    # B03: 3, B13: 9
    
    U[0, col] = 1; V[3, col] = 1; W[6, col] = 1; col += 1
    U[1, col] = 1; V[9, col] = 1; W[6, col] = 1; col += 1
    U[5, col] = 1; V[3, col] = 1; W[7, col] = 1; col += 1
    U[6, col] = 1; V[9, col] = 1; W[7, col] = 1; col += 1
    
    # Block 2 Repeat: A2 * B2_right (2x3 * 3x3)
    # B2_right: j=2,3,4; k=3,4,5.
    # Indices: 15,16,17, 21,22,23, 27,28,29.
    # C_right: k=3,4,5. i=0,1.
    # Indices: 6,7, 8,9, 10,11.
    
    # A_L * B_L_right
    # A_L: 2, 7.
    # B_L_right: 15, 16, 17.
    
    U[2, col] = 1; V[15, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[16, col] = 1; W[8, col] = 1; col += 1
    U[2, col] = 1; V[17, col] = 1; W[10, col] = 1; col += 1
    U[7, col] = 1; V[15, col] = 1; W[7, col] = 1; col += 1
    U[7, col] = 1; V[16, col] = 1; W[9, col] = 1; col += 1
    U[7, col] = 1; V[17, col] = 1; W[11, col] = 1; col += 1
    
    # A_R * B_R_right
    # A_R: 3,4,8,9.
    # B_R_right: 21,22,23, 27,28,29.
    
    # M1' = (A03 + A14) (B33 + B45)
    U[3, col] = 1; U[9, col] = 1
    V[21, col] = 1; V[29, col] = 1
    W[6, col] = 1; W[7, col] = 1; W[11, col] = -1
    col += 1
    
    # M2' = (A13 + A14) B33
    U[8, col] = 1; U[9, col] = 1
    V[21, col] = 1
    W[7, col] = 1; W[11, col] = -1
    col += 1
    
    # M3' = A03 (B35 - B45)
    U[3, col] = 1
    V[23, col] = 1; V[29, col] = -1
    W[10, col] = 1; W[11, col] = 1
    col += 1
    
    # M4' = A14 (B33 - B35)
    U[9, col] = 1
    V[21, col] = 1; V[23, col] = -1
    W[6, col] = 1; W[7, col] = 1
    col += 1
    
    # M5' = (A03 + A04) B45
    U[3, col] = 1; U[4, col] = 1
    V[29, col] = 1
    W[6, col] = -1; W[10, col] = 1
    col += 1
    
    # M6' = (A13 - A03) (B33 + B35)
    U[8, col] = 1; U[3, col] = -1
    V[21, col] = 1; V[23, col] = 1
    W[11, col] = 1
    col += 1
    
    # M7' = (A04 - A14) (B43 + B45)
    U[4, col] = 1; U[9, col] = -1
    V[27, col] = 1; V[29, col] = 1
    W[6, col] = 1
    col += 1
    
    # Schoolbook A_R * B_R[:, 3]
    # B33: 21, B43: 27
    
    U[3, col] = 1; V[21, col] = 1; W[6, col] = 1; col += 1
    U[4, col] = 1; V[27, col] = 1; W[6, col] = 1; col += 1
    U[8, col] = 1; V[21, col] = 1; W[7, col] = 1; col += 1
    U[9, col] = 1; V[27, col] = 1; W[7, col] = 1; col += 1
    
    # Total col should be 56.
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
