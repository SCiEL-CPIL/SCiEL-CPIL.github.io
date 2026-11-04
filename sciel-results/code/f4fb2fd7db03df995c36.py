"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Uses Strassen for the 4x4x4 part (rank 49) and schoolbook for the 4x4x3 part (rank 48).
    Total rank: 97
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Split p into 4 + 3
    p1, p2 = 4, 3
    
    # Part 1: 4x4 x 4x4 (Strassen, rank 49)
    U1, V1, W1 = _strassen_4x4x4()
    
    # Part 2: 4x4 x 4x3 (Schoolbook, rank 48)
    U2, V2, W2 = _schoolbook_4x4x3()
    
    # Concatenate
    rank = len(U1)
    U = [row + [0]*len(U2) for row in U1]
    for row in U2:
        U.append(row)
    
    V = [row + [0]*len(V2) for row in V1]
    for row in V2:
        V.append(row)
    
    W = [row + [0]*len(W2) for row in W1]
    for row in W2:
        W.append(row)
    
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, c] = coefficient of A[i,j] in factor c
    V[j*2+k, c] = coefficient of B[j,k] in factor c  
    W[k*2+i, c] = coefficient of C[i,k] in factor c
    
    Strassen formulas:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 - M3 + M4 - M6
    C01 = M3 - M1 + M5 + M7
    C10 = M2 + M4 - M7 + M6
    C11 = M1 - M2 + M3 + M5
    """
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=4, (0,1)=5, (1,0)=6, (1,1)=7
    # C indices: (0,0)=8, (0,1)=9, (1,0)=10, (1,1)=11
    
    # Each factor c has coefficients for A entries and B entries
    # U[c] = coefficients for A (4 entries)
    # V[c] = coefficients for B (4 entries)
    # W[c] = coefficients for C (4 entries)
    
    # M1 = (A00+A11)(B00+B11)
    M1_U = [1, 0, 0, 1]  # A00 + A11
    M1_V = [1, 0, 0, 1]  # B00 + B11
    M1_W = [1, 0, 0, 0]  # contributes to C00 with +1
    M1_W += [0, 0, 0, 0]  # but we need to track which C entry
    
    # Let me redo this more carefully
    # For each Strassen factor, I need:
    # - Linear combination of A entries (U coefficients)
    # - Linear combination of B entries (V coefficients)  
    # - Which C entry it contributes to and with what coefficient (W)
    
    # Actually, W should map the factor to C entries
    # Let's define: factor_c = sum_i U[i,c] * A[i] * sum_j V[j,c] * B[j]
    # And C[k] = sum_c W[k,c] * factor_c
    
    # M1 = (A00+A11)(B00+B11)
    M1_U = [1, 0, 0, 1]  # A[0]+A[3]
    M1_V = [1, 0, 0, 1]  # B[0]+B[3]
    # M1 contributes: C00 += 1*M1, C11 += 1*M1
    M1_W = {8: 1, 11: 1}  # C00=8, C11=11
    
    # M2 = (A10+A11)B00
    M2_U = [0, 0, 1, 1]  # A[2]+A[3]
    M2_V = [1, 0, 0, 0]  # B[0]
    # M2 contributes: C10 += 1*M2, C11 += -1*M2
    M2_W = {10: 1, 11: -1}
    
    # M3 = A00(B01-B11)
    M3_U = [1, 0, 0, 0]  # A[0]
    M3_V = [0, 1, 0, -1]  # B[1]-B[3]
    # M3 contributes: C00 += -1*M3, C01 += 1*M3, C11 += 1*M3
    M3_W = {8: -1, 9: 1, 11: 1}
    
    # M4 = A11(B10-B00)
    M4_U = [0, 0, 0, 1]  # A[3]
    M4_V = [0, 0, 1, -1]  # B[2]-B[0]
    # M4 contributes: C00 += 1*M4, C10 += 1*M4
    M4_W = {8: 1, 10: 1}
    
    # M5 = (A00+A01)B11
    M5_U = [1, 1, 0, 0]  # A[0]+A[1]
    M5_V = [0, 0, 0, 1]  # B[3]
    # M5 contributes: C01 += 1*M5, C11 += 1*M5
    M5_W = {9: 1, 11: 1}
    
    # M6 = (A10-A00)(B00+B01)
    M6_U = [-1, 0, 1, 0]  # -A[0]+A[2]
    M6_V = [1, 1, 0, 0]  # B[0]+B[1]
    # M6 contributes: C00 += -1*M6, C10 += 1*M6
    M6_W = {8: -1, 10: 1}
    
    # M7 = (A01-A11)(B10+B11)
    M7_U = [0, 1, 0, -1]  # A[1]-A[3]
    M7_V = [0, 0, 1, 1]  # B[2]+B[3]
    # M7 contributes: C01 += 1*M7, C10 += -1*M7
    M7_W = {9: 1, 10: -1}
    
    factors = [
        (M1_U, M1_V, M1_W),
        (M2_U, M2_V, M2_W),
        (M3_U, M3_V, M3_W),
        (M4_U, M4_V, M4_W),
        (M5_U, M5_V, M5_W),
        (M6_U, M6_V, M6_W),
        (M7_U, M7_V, M7_W),
    ]
    
    # Build U (4x7), V (4x7), W (4x7)
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    for c, (Uf, Vf, Wf) in enumerate(factors):
        for i in range(4):
            U[i][c] = Uf[i]
            V[i][c] = Vf[i]
        for k, coeff in Wf.items():
            # k is C index: 8=C00, 9=C01, 10=C10, 11=C11
            # C index in our scheme: k*n+i where n=2
            # C00: k=0,i=0 -> 0*2+0=0
            # C01: k=1,i=0 -> 1*2+0=2  
            # Wait, let me re-read the convention
            # W[k*n+i, r] - so row index is k*n+i
            # For 2x2 output: rows are 0,1,2,3 corresponding to (k,i)
            # C[i,k] in standard notation, but here W[k*n+i]
            # So W row = k*2 + i
            # C00: i=0,k=0 -> row 0*2+0 = 0
            # C01: i=0,k=1 -> row 1*2+0 = 2
            # C10: i=1,k=0 -> row 0*2+1 = 1
            # C11: i=1,k=1 -> row 1*2+1 = 3
            
            # My Wf keys were: 8=C00, 9=C01, 10=C10, 11=C11
            # Map: 8->0, 9->2, 10->1, 11->3
            row_map = {8: 0, 9: 2, 10: 1, 11: 3}
            W[row_map[k]][c] = coeff
    
    return U, V, W


def _strassen_4x4x4():
    """
    Strassen decomposition for 4x4 x 4x4 matrix multiplication (rank 49).
    Uses recursive Strassen: split into 2x2 blocks, apply Strassen to each.
    """
    # Split 4x4 into four 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each Ai_j is 2x2
    # B = [[B00, B01], [B10, B11]] where each B_i_j is 2x2
    # C = [[C00, C01], [C10, C11]] where each C_i_j is 2x2
    
    # Apply Strassen at the block level:
    # M1 = (A00+A11)(B00+B11)  -> 2x2 x 2x2 = use Strassen (rank 7)
    # M2 = (A10+A11)B00        -> 2x2 x 2x2 = use Strassen (rank 7)
    # M3 = A00(B01-B11)        -> 2x2 x 2x2 = use Strassen (rank 7)
    # M4 = A11(B10-B00)        -> 2x2 x 2x2 = use Strassen (rank 7)
    # M5 = (A00+A01)B11        -> 2x2 x 2x2 = use Strassen (rank 7)
    # M6 = (A10-A00)(B00+B01)  -> 2x2 x 2x2 = use Strassen (rank 7)
    # M7 = (A01-A11)(B10+B11)  -> 2x2 x 2x2 = use Strassen (rank 7)
    # Total: 7*7 = 49 factors
    
    # For each block-level Strassen factor, we need to:
    # 1. Form the linear combination of A blocks
    # 2. Form the linear combination of B blocks
    # 3. Multiply them using Strassen 2x2x2 (gives 7 sub-factors)
    
    # Let's define the block-level Strassen formulas
    # Block A indices: A00=0, A01=1, A10=2, A11=3
    # Block B indices: B00=4, B01=5, B10=6, B11=7
    
    block_formulas = [
        # (A_blocks, B_blocks, C_block_output, C_coeff)
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {4: 1, 7: 1}, {0: 1, 3: 1}),  # C00 += 1*M1, C11 += 1*M1
        # M2 = (A10+A11)B00
        ({2: 1, 3: 1}, {4: 1}, {2: 1, 3: -1}),  # C10 += 1*M2, C11 += -1*M2
        # M3 = A00(B01-B11)
        ({0: 1}, {5: 1, 7: -1}, {0: -1, 1: 1, 3: 1}),  # C00 += -1*M3, C01 += 1*M3, C11 += 1*M3
        # M4 = A11(B10-B00)
        ({3: 1}, {6: 1, 4: -1}, {0: 1, 2: 1}),  # C00 += 1*M4, C10 += 1*M4
        # M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {7: 1}, {1: 1, 3: 1}),  # C01 += 1*M5, C11 += 1*M5
        # M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {4: 1, 5: 1}, {0: -1, 2: 1}),  # C00 += -1*M6, C10 += 1*M6
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {6: 1, 7: 1}, {1: 1, 2: -1}),  # C01 += 1*M7, C10 += -1*M7
    ]
    
    # Get the 2x2x2 Strassen decomposition
    U22, V22, W22 = _strassen_2x2x2()
    
    # Build the full 4x4x4 decomposition
    # A has 16 entries: A[i,j] where i,j in 0..3
    # Block A00: A[0,0],A[0,1],A[1,0],A[1,1] -> indices 0,1,2,3
    # Block A01: A[0,2],A[0,3],A[1,2],A[1,3] -> indices 4,5,6,7
    # Block A10: A[2,0],A[2,1],A[3,0],A[3,1] -> indices 8,9,10,11
    # Block A11: A[2,2],A[2,3],A[3,2],A[3,3] -> indices 12,13,14,15
    
    # Similarly for B
    # B has 16 entries: B[j,k] where j,k in 0..3
    # Block B00: B[0,0],B[0,1],B[1,0],B[1,1] -> indices 0,1,2,3
    # Block B01: B[0,2],B[0,3],B[1,2],B[1,3] -> indices 4,5,6,7
    # Block B10: B[2,0],B[2,1],B[3,0],B[3,1] -> indices 8,9,10,11
    # Block B11: B[2,2],B[2,3],B[3,2],B[3,3] -> indices 12,13,14,15
    
    # C has 16 entries: C[i,k] where i,k in 0..3
    # Block C00: C[0,0],C[0,1],C[1,0],C[1,1] -> indices 0,1,2,3
    # Block C01: C[0,2],C[0,3],C[1,2],C[1,3] -> indices 4,5,6,7
    # Block C10: C[2,0],C[2,1],C[3,0],C[3,1] -> indices 8,9,10,11
    # Block C11: C[2,2],C[2,3],C[3,2],C[3,3] -> indices 12,13,14,15
    
    # For each of the 7 block-level factors, we have 7 sub-factors from Strassen 2x2x2
    # Total: 49 factors
    
    U_full = [[0]*49 for _ in range(16)]  # 16 A entries x 49 factors
    V_full = [[0]*49 for _ in range(16)]  # 16 B entries x 49 factors
    W_full = [[0]*49 for _ in range(16)]  # 16 C entries x 49 factors
    
    for block_idx, (A_blocks, B_blocks, C_blocks) in enumerate(block_formulas):
        for sub_idx in range(7):
            factor_idx = block_idx * 7 + sub_idx
            
            # For this factor, the A coefficient is:
            # For each A block b in A_blocks with coeff ca:
            #   For each entry within that block, use U22[entry, sub_idx] * ca
            for b_idx, ca in A_blocks.items():
                # b_idx: 0=A00, 1=A01, 2=A10, 3=A11
                base_a = b_idx * 4
                for entry in range(4):
                    U_full[base_a + entry][factor_idx] = ca * U22[entry][sub_idx]
            
            # Similarly for B
            for b_idx, cb in B_blocks.items():
                # b_idx: 4=B00, 5=B01, 6=B10, 7=B11
                b_idx_local = b_idx - 4  # 0,1,2,3
                base_b = b_idx_local * 4
                for entry in range(4):
                    V_full[base_b + entry][factor_idx] = cb * V22[entry][sub_idx]
            
            # For C output
            for c_idx, cc in C_blocks.items():
                # c_idx: 0=C00, 1=C01, 2=C10, 3=C11
                base_c = c_idx * 4
                for entry in range(4):
                    W_full[base_c + entry][factor_idx] = cc * W22[entry][sub_idx]
    
    return U_full, V_full, W_full


def _schoolbook_4x4x3():
    """
    Schoolbook decomposition for 4x4 x 4x3 matrix multiplication (rank 48).
    """
    n, m, p = 4, 4, 3
    rank = n * m * p  # 48
    
    U = [[0]*rank for _ in range(n*m)]  # 16 x 48
    V = [[0]*rank for _ in range(m*p)]  # 12 x 48
    W = [[0]*rank for _ in range(p*n)]  # 12 x 48
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][column] = 1
                V[j*p + k][column] = 1
                W[k*n + i][column] = 1
                column += 1
    
    return U, V, W
# EVOLVE-BLOCK-END

