"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use recursive Strassen: rank 7^2 = 49
    rank = 49
    
    # Initialize
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen for 2x2: the 7 multiplication patterns
    # For matrices [[a,b],[c,d]] and [[e,f],[g,h]]:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # We'll build the 4x4 decomposition by composing two levels of Strassen.
    # Level 1: 4x4 = 2x2 blocks of 2x2
    # Level 2: Each 2x2 product uses 7 scalar multiplications
    
    # Let's define the Strassen decomposition for 2x2 explicitly.
    # For 2x2 matrix multiplication, the tensor decomposition has rank 7.
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c] for i,j,k in {0,1}
    
    # Strassen 2x2 decomposition:
    # The 7 terms correspond to:
    # term 0: (a+d)(e+h) -> U: a=1,d=1; V: e=1,h=1; W: contributes to C11,C22
    # term 1: (c+d)e -> U: c=1,d=1; V: e=1; W: contributes to C21
    # term 2: a(f-h) -> U: a=1; V: f=1,h=-1; W: contributes to C12
    # term 3: d(g-e) -> U: d=1; V: g=1,e=-1; W: contributes to C11
    # term 4: (a+b)h -> U: a=1,b=1; V: h=1; W: contributes to C11,C12
    # term 5: (c-a)(e+f) -> U: c=1,a=-1; V: e=1,f=1; W: contributes to C11
    # term 6: (b-d)(g+h) -> U: b=1,d=-1; V: g=1,h=1; W: contributes to C22
    
    # Let me define the 2x2 Strassen decomposition explicitly.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # U2 has shape (4, 7), V2 has shape (4, 7), W2 has shape (4, 7)
    # Index: U2[i*2+j], V2[j*2+k], W2[k*2+i]
    
    # Term 0: (a+d)(e+h)
    # U2: a=1 (index 0), d=1 (index 3)
    # V2: e=1 (index 0), h=1 (index 3)
    # W2: This term contributes to C11 and C22.
    # C11 = ae+bg, C22 = cf+dh
    # (a+d)(e+h) = ae + ah + de + dh
    # C11 gets: ae (from a*e), but we need to combine terms.
    # Actually, the standard Strassen:
    # C11 = m1 - m4 + m5 + m6
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m3 + m2 + m7
    
    # So W2 for each term indicates which C entries it contributes to, with coefficients.
    # W2[k*2+i, c] where k is row of C, i is col of C.
    # C indices: C[0,0]=m1-m4+m5+m6, C[0,1]=m3+m5, C[1,0]=m2+m4, C[1,1]=m1-m3+m2+m7
    
    # W2[0*2+0] = W2[0] for C[0,0]
    # W2[0*2+1] = W2[1] for C[0,1]
    # W2[1*2+0] = W2[2] for C[1,0]
    # W2[1*2+1] = W2[3] for C[1,1]
    
    # Term 0 (m1): contributes +1 to C[0,0] and +1 to C[1,1]
    # W2[0]=1, W2[1]=0, W2[2]=0, W2[3]=1
    # Term 1 (m2): contributes +1 to C[1,0] and +1 to C[1,1]
    # W2[0]=0, W2[1]=0, W2[2]=1, W2[3]=1
    # Term 2 (m3): contributes +1 to C[0,1] and -1 to C[1,1]
    # W2[0]=0, W2[1]=1, W2[2]=0, W2[3]=-1
    # Term 3 (m4): contributes -1 to C[0,0] and +1 to C[1,0]
    # W2[0]=-1, W2[1]=0, W2[2]=1, W2[3]=0
    # Term 4 (m5): contributes +1 to C[0,0] and +1 to C[0,1]
    # W2[0]=1, W2[1]=1, W2[2]=0, W2[3]=0
    # Term 5 (m6): contributes +1 to C[0,0]
    # W2[0]=1, W2[1]=0, W2[2]=0, W2[3]=0
    # Term 6 (m7): contributes +1 to C[1,1]
    # W2[0]=0, W2[1]=0, W2[2]=0, W2[3]=1
    
    # Now for U2 and V2:
    # U2[i*2+j, c] for the A matrix entries
    # A = [[a, b], [c, d]] -> indices: a=0, b=1, c=2, d=3
    
    # Term 0: (a+d) -> U2[0]=1, U2[3]=1
    # Term 1: (c+d) -> U2[2]=1, U2[3]=1
    # Term 2: a -> U2[0]=1
    # Term 3: d -> U2[3]=1
    # Term 4: (a+b) -> U2[0]=1, U2[1]=1
    # Term 5: (c-a) -> U2[2]=1, U2[0]=-1
    # Term 6: (b-d) -> U2[1]=1, U2[3]=-1
    
    # V2[j*2+k, c] for the B matrix entries
    # B = [[e, f], [g, h]] -> indices: e=0, f=1, g=2, h=3
    
    # Term 0: (e+h) -> V2[0]=1, V2[3]=1
    # Term 1: e -> V2[0]=1
    # Term 2: (f-h) -> V2[1]=1, V2[3]=-1
    # Term 3: (g-e) -> V2[2]=1, V2[0]=-1
    # Term 4: h -> V2[3]=1
    # Term 5: (e+f) -> V2[0]=1, V2[1]=1
    # Term 6: (g+h) -> V2[2]=1, V2[3]=1
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 0
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    # Term 1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = 1
    # Term 2
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = -1
    # Term 3
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = -1; W2[2][3] = 1
    # Term 4
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = 1; W2[1][4] = 1
    # Term 5
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[0][5] = 1
    # Term 6
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[3][6] = 1
    
    # Now compose: 4x4 = 2x2 blocks of 2x2
    # A4 = [[A11, A12], [A21, A22]] where each is 2x2
    # A11 uses A indices 0,1,4,5 (a=0,b=1,c=4,d=5)
    # A12 uses A indices 2,3,6,7
    # A21 uses A indices 8,9,12,13
    # A22 uses A indices 10,11,14,15
    
    # Similarly for B4.
    
    # The 4x4 Strassen at block level:
    # S1 = A11 + A22, S2 = A21 + A22, S3 = A11 + A12, S4 = A22 + A12
    # T1 = B11 + B22, T2 = B11 + B12, T3 = B21 + B22, T4 = B22 + B12
    # M1 = S1*T1, M2 = S2*B11, M3 = A11*T2, M4 = A22*T3, M5 = S3*B22, M6 = A12*T4, M7 = S4*B21
    # C11 = M1 - M4 + M5 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M7
    
    # Each M is a 2x2 matrix product, decomposed with rank 7.
    # So total rank = 7*7 = 49.
    
    # For each block-level term (7 terms) and each inner Strassen term (7 terms):
    # We need to combine the U, V, W factors.
    
    # Let me define the block indices:
    # A block (row, col) of 2x2 blocks:
    # A11: rows 0-1, cols 0-1 -> A indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> A indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> A indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> A indices 10,11,14,15
    
    # B block:
    # B11: rows 0-1, cols 0-1 -> B indices 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> B indices 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> B indices 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> B indices 10,11,14,15
    
    # C block:
    # C11: rows 0-1, cols 0-1 -> C indices 0,1,4,5
    # C12: rows 0-1, cols 2-3 -> C indices 2,3,6,7
    # C21: rows 2-3, cols 0-1 -> C indices 8,9,12,13
    # C22: rows 2-3, cols 2-3 -> C indices 10,11,14,15
    
    # For a 2x2 matrix product M = X*Y where X is a linear combination of A blocks
    # and Y is a linear combination of B blocks, the U factor for the 4x4 decomposition
    # is the product of the block-level coefficient and the inner Strassen U factor.
    
    # Let's define the 7 block-level terms with their X and Y:
    # Term 0: X = A11+A22, Y = B11+B22
    # Term 1: X = A21+A22, Y = B11
    # Term 2: X = A11, Y = B11+B12
    # Term 3: X = A22, Y = B21+B22
    # Term 4: X = A11+A12, Y = B22
    # Term 5: X = A12, Y = B22+B12
    # Term 6: X = A22+A12, Y = B21
    
    # And the W factors for the block-level:
    # C11 = M0 - M3 + M4 + M5
    # C12 = M2 + M4
    # C21 = M1 + M3
    # C22 = M0 - M2 + M1 + M6
    
    # So for block term t, the contribution to C blocks:
    # w_block[t][block] = coefficient
    # w_block[0] = {C11:1, C12:0, C21:0, C22:1}
    # w_block[1] = {C11:0, C12:0, C21:1, C22:1}
    # w_block[2] = {C11:0, C12:1, C21:0, C22:-1}
    # w_block[3] = {C11:-1, C12:0, C21:1, C22:0}
    # w_block[4] = {C11:1, C12:1, C21:0, C22:0}
    # w_block[5] = {C11:1, C12:0, C21:0, C22:0}
    # w_block[6] = {C11:0, C12:0, C21:0, C22:1}
    
    # Now for each block term t (0-6) and inner term s (0-6):
    # The combined column index is t*7 + s
    # U[4*i+j, t*7+s] = (block_coeff for A in term t) * U2[inner_idx, s]
    # V[4*j+k, t*7+s] = (block_coeff for B in term t) * V2[inner_idx, s]
    # W[4*k+i, t*7+s] = (block_coeff for C in term t) * W2[inner_idx, s]
    
    # Define block coefficients:
    # For term t, X is a linear combination of A blocks.
    # x_block[t][block] = coefficient of A_block in X for term t
    # y_block[t][block] = coefficient of B_block in Y for term t
    
    x_block = [
        {0:1, 1:0, 2:0, 3:1},  # term 0: A11+A22 (blocks: 0=A11,1=A12,2=A21,3=A22)
        {0:0, 1:0, 2:1, 3:1},  # term 1: A21+A22
        {0:1, 1:0, 2:0, 3:0},  # term 2: A11
        {0:0, 1:0, 2:0, 3:1},  # term 3: A22
        {0:1, 1:1, 2:0, 3:0},  # term 4: A11+A12
        {0:0, 1:1, 2:0, 3:0},  # term 5: A12
        {0:0, 1:1, 2:0, 3:1},  # term 6: A22+A12
    ]
    
    y_block = [
        {0:1, 1:0, 2:0, 3:1},  # term 0: B11+B22
        {0:1, 1:0, 2:0, 3:0},  # term 1: B11
        {0:1, 1:1, 2:0, 3:0},  # term 2: B11+B12
        {0:0, 1:0, 2:1, 3:1},  # term 3: B21+B22
        {0:0, 1:0, 2:0, 3:1},  # term 4: B22
        {0:0, 1:1, 2:0, 3:1},  # term 5: B22+B12
        {0:0, 1:0, 2:1, 3:0},  # term 6: B21
    ]
    
    w_block = [
        {0:1, 1:0, 2:0, 3:1},  # term 0: C11+C22
        {0:0, 1:0, 2:1, 3:1},  # term 1: C21+C22
        {0:0, 1:1, 2:0, 3:-1}, # term 2: C12-C22
        {0:-1, 1:0, 2:1, 3:0}, # term 3: -C11+C21
        {0:1, 1:1, 2:0, 3:0},  # term 4: C11+C12
        {0:1, 1:0, 2:0, 3:0},  # term 5: C11
        {0:0, 1:0, 2:0, 3:1},  # term 6: C22
    ]
    
    # Block to index mapping:
    # Block 0 (A11): A indices [0,1,4,5] -> local [0,1,2,3]
    # Block 1 (A12): A indices [2,3,6,7] -> local [0,1,2,3]
    # Block 2 (A21): A indices [8,9,12,13] -> local [0,1,2,3]
    # Block 3 (A22): A indices [10,11,14,15] -> local [0,1,2,3]
    
    block_a_indices = [
        [0, 1, 4, 5],   # block 0
        [2, 3, 6, 7],   # block 1
        [8, 9, 12, 13], # block 2
        [10, 11, 14, 15] # block 3
    ]
    
    block_b_indices = [
        [0, 1, 4, 5],   # block 0
        [2, 3, 6, 7],   # block 1
        [8, 9, 12, 13], # block 2
        [10, 11, 14, 15] # block 3
    ]
    
    block_c_indices = [
        [0, 1, 4, 5],   # block 0
        [2, 3, 6, 7],   # block 1
        [8, 9, 12, 13], # block 2
        [10, 11, 14, 15] # block 3
    ]
    
    # Now build U, V, W
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            # U factor
            for block in range(4):
                coeff = x_block[t][block]
                if coeff == 0:
                    continue
                for local in range(4):
                    u_val = U2[local][s]
                    if u_val == 0:
                        continue
                    a_idx = block_a_indices[block][local]
                    U[a_idx][col] += coeff * u_val
            
            # V factor
            for block in range(4):
                coeff = y_block[t][block]
                if coeff == 0:
                    continue
                for local in range(4):
                    v_val = V2[local][s]
                    if v_val == 0:
                        continue
                    b_idx = block_b_indices[block][local]
                    V[b_idx][col] += coeff * v_val
            
            # W factor
            for block in range(4):
                coeff = w_block[t][block]
                if coeff == 0:
                    continue
                for local in range(4):
                    w_val = W2[local][s]
                    if w_val == 0:
                        continue
                    c_idx = block_c_indices[block][local]
                    W[c_idx][col] += coeff * w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
