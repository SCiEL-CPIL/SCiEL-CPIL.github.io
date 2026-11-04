"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # A: n x m = 2x4, B: m x p = 4x5, C: n x p = 2x5
    # W index convention: W[k*n + i, r]
    
    # Block structure:
    # Block A: j in {0,1}, k in {0,1,2} -> 2x2x3, rank 11
    # Block B: j in {0,1}, k in {3,4}   -> 2x2x2, rank 7
    # Block C: j in {2,3}, k in {0,1,2} -> 2x2x3, rank 11
    # Block D: j in {2,3}, k in {3,4}   -> 2x2x2, rank 7
    # Total rank: 11 + 7 + 11 + 7 = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]   # 8 x 36
    V = [[0] * rank for _ in range(m * p)]   # 20 x 36
    W = [[0] * rank for _ in range(p * n)]   # 10 x 36
    
    # Strassen 2x2x2 factors (local indices)
    # U_local: 4x7, V_local: 4x7, W_local: 4x7
    # A rows: 0=A00, 1=A01, 2=A10, 3=A11
    # B rows: 0=B00, 1=B01, 2=B10, 3=B11
    # C/W rows: 0=C00, 1=C10, 2=C01, 3=C11 (W[k*2+i])
    
    U_strassen = [
        [1,  0,  1,  0,  1, -1,  0],  # A00+A11, A10+A11, A00, A11, A00+A10, A10-A00, A01-A11
        [0,  0,  0,  0,  0,  0,  1],
        [0,  1,  0,  0,  1,  1, -1],
        [1,  1,  0,  1,  0,  0,  0],
    ]
    
    V_strassen = [
        [1,  1,  0,  1,  0,  1,  0],  # B00+B11, B00, B10+B11, B00, B11, B00+B01, B10+B11
        [0,  0,  0,  0,  0,  1,  0],
        [0,  0,  1,  0,  0,  0,  1],
        [1,  0,  1,  0,  1,  0,  1],
    ]
    
    W_strassen = [
        [1,  0,  0, -1,  0,  1,  0],  # C00
        [0,  1,  0,  0,  1,  0,  0],  # C10
        [0,  0,  1,  0,  0,  0,  1],  # C01
        [1,  0, -1,  1,  0,  0,  0],  # C11
    ]
    
    # 2x2x3 factors: rank 11
    # U_223: 4x11 (cols 0-6 Strassen, cols 7-10 schoolbook for col 2)
    U_223 = [
        [1,  0,  1,  0,  1, -1,  0,  1,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0],
        [0,  1,  0,  0,  1,  1, -1,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  1],
    ]
    
    # V_223: 6x11 (B is 2x3, rows: 0=B00,1=B01,2=B02,3=B10,4=B11,5=B12)
    V_223 = [
        [1,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0],  # B00
        [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],  # B01
        [0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],  # B02
        [0,  0,  1,  0,  0,  0,  1,  0,  1,  0,  0],  # B10
        [1,  0,  1,  0,  1,  0,  1,  0,  0,  1,  0],  # B11
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],  # B12
    ]
    
    # W_223: 6x11 (C is 2x3, W index = k*2+i: 0=C00,1=C10,2=C01,3=C11,4=C02,5=C12)
    W_223 = [
        [1,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0],  # C00
        [0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0],  # C10
        [0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0],  # C01
        [1,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0],  # C11
        [0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],  # C02
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],  # C12
    ]
    
    # Now assemble the full 2x4x5 decomposition
    # Block A: j in {0,1}, k in {0,1,2}, rank 11, cols 0-10
    #   A rows: i*4+j -> (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    #   B rows: j*5+k -> (0,0)->0,(0,1)->1,(0,2)->2,(1,0)->5,(1,1)->6,(1,2)->7
    #   W rows: k*2+i -> 0,1,2,3,4,5
    
    a_rows_A = [0, 1, 4, 5]  # local 0,1,2,3 -> global
    b_rows_A = [0, 1, 2, 5, 6, 7]
    w_rows_A = [0, 1, 2, 3, 4, 5]
    
    for r in range(11):
        for lr in range(4):
            U[a_rows_A[lr]][r] += U_223[lr][r]
        for lr in range(6):
            V[b_rows_A[lr]][r] += V_223[lr][r]
        for lr in range(6):
            W[w_rows_A[lr]][r] += W_223[lr][r]
    
    # Block B: j in {0,1}, k in {3,4}, rank 7, cols 11-17
    #   A rows: same as Block A: 0,1,4,5
    #   B rows: (0,3)->3,(0,4)->4,(1,3)->8,(1,4)->9
    #   W rows: k*2+i for k=3,4: 6,7,8,9
    
    a_rows_B = [0, 1, 4, 5]
    b_rows_B = [3, 4, 8, 9]
    w_rows_B = [6, 7, 8, 9]
    
    for r in range(7):
        for lr in range(4):
            U[a_rows_B[lr]][11 + r] += U_strassen[lr][r]
        for lr in range(4):
            V[b_rows_B[lr]][11 + r] += V_strassen[lr][r]
        for lr in range(4):
            W[w_rows_B[lr]][11 + r] += W_strassen[lr][r]
    
    # Block C: j in {2,3}, k in {0,1,2}, rank 11, cols 18-28
    #   A rows: (0,2)->2,(0,3)->3,(1,2)->6,(1,3)->7
    #   B rows: (2,0)->10,(2,1)->11,(2,2)->12,(3,0)->15,(3,1)->16,(3,2)->17
    #   W rows: same as Block A: 0,1,2,3,4,5
    
    a_rows_C = [2, 3, 6, 7]
    b_rows_C = [10, 11, 12, 15, 16, 17]
    w_rows_C = [0, 1, 2, 3, 4, 5]
    
    for r in range(11):
        for lr in range(4):
            U[a_rows_C[lr]][18 + r] += U_223[lr][r]
        for lr in range(6):
            V[b_rows_C[lr]][18 + r] += V_223[lr][r]
        for lr in range(6):
            W[w_rows_C[lr]][18 + r] += W_223[lr][r]
    
    # Block D: j in {2,3}, k in {3,4}, rank 7, cols 29-35
    #   A rows: same as Block C: 2,3,6,7
    #   B rows: (2,3)->13,(2,4)->14,(3,3)->18,(3,4)->19
    #   W rows: same as Block B: 6,7,8,9
    
    a_rows_D = [2, 3, 6, 7]
    b_rows_D = [13, 14, 18, 19]
    w_rows_D = [6, 7, 8, 9]
    
    for r in range(7):
        for lr in range(4):
            U[a_rows_D[lr]][29 + r] += U_strassen[lr][r]
        for lr in range(4):
            V[b_rows_D[lr]][29 + r] += V_strassen[lr][r]
        for lr in range(4):
            W[w_rows_D[lr]][29 + r] += W_strassen[lr][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
