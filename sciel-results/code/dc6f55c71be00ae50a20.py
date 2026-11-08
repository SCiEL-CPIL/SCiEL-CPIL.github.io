"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Kronecker product of known rank-minimal decompositions:
    #   M_{2,2,2} has rank 7 (Strassen)
    #   M_{2,2,3} has rank 11 (known)
    #   M_{4,4,6} = M_{2,2,3} ⊗ M_{2,2,2}  → rank 11 * 7 = 77
    #
    # We build the tensor factors U, V, W for M_{4,4,6} as Kronecker products
    # of the corresponding factors for M_{2,2,3} and M_{2,2,2}.
    
    # --- Strassen M_{2,2,2}, rank 7 ---
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U_S = [
        [1,  0,  0,  0,  0,  0,  0],
        [0,  1,  0,  0,  0,  0,  0],
        [0,  0,  1,  0,  0,  0,  0],
        [0,  0,  0,  1,  0,  0,  0],
    ]
    V_S = [
        [0,  0,  0,  0,  1,  0,  0],
        [0,  0,  0,  0,  0,  1,  0],
        [0,  0,  0,  0,  0,  0,  1],
        [0,  0,  0,  0,  0,  0,  0],
    ]
    W_S = [
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0],
    ]
    # Let me use the standard Strassen decomposition properly.
    # C = AB where A,B are 2x2.
    # m1 = (A11+A22)(B11+B22)
    # m2 = (A21+A22)B11
    # m3 = A11(B21-B22)
    # m4 = A22(B12-B11)
    # m5 = (A11+A12)B22
    # m6 = (A21-A11)(B11+B12)
    # m7 = (A12-A22)(B21+B22)
    # C11 = m1+m4-m5-m7
    # C12 = m3+m5
    # C21 = m2+m4
    # C22 = m1-m2+m3-m6
    #
    # U rows indexed by (i,j) -> i*2+j (i=row of A, j=col of A)
    # V rows indexed by (j,k) -> j*2+k (j=row of B = col of A, k=col of B)
    # W rows indexed by (k,i) -> k*2+i (k=col of C = col of B, i=row of C = row of A)
    
    U_S = [
        [1,  1,  0,  0,  1,  0,  0],  # A11: m1, m5
        [0,  0,  1,  0,  0,  0,  1],  # A12: m3, m7  -> wait, m3 uses A11, m7 uses A12
        [1,  1,  0,  0,  0,  1,  0],  # A21: m1, m2, m6
        [1,  1,  0,  1,  0,  0,  0],  # A22: m1, m2, m4
    ]
    # Let me redo carefully.
    # m1 = (A11+A22)(B11+B22): U entries: A11→1, A22→1; V: B11→1, B22→1
    # m2 = (A21+A22)B11: U: A21→1, A22→1; V: B11→1
    # m3 = A11(B21-B22): U: A11→1; V: B21→1, B22→-1
    # m4 = A22(B12-B11): U: A22→1; V: B12→1, B11→-1
    # m5 = (A11+A12)B22: U: A11→1, A12→1; V: B22→1
    # m6 = (A21-A11)(B11+B12): U: A21→1, A11→-1; V: B11→1, B12→1
    # m7 = (A12-A22)(B21+B22): U: A12→1, A22→-1; V: B21→1, B22→1
    #
    # U rows: [A11, A12, A21, A22] (index 0,1,2,3)
    # V rows: [B11, B12, B21, B22] (index 0,1,2,3)
    # W rows: [C11, C12, C21, C22] (index 0,1,2,3), W[k*2+i] so W[0]=C11, W[1]=C12, W[2]=C21, W[3]=C22
    
    U_S = [
        [1,  0,  1,  0,  1, -1,  0],   # A11: m1,m3,m5,m6(-1)
        [0,  1,  0,  0,  1,  0,  1],   # A12: m3(0),m5(1),m7(1) -> m3 doesn't use A12
        [1,  0,  1,  0,  0,  1,  0],   # A21: m1,m2,m6
        [1,  0,  0,  1,  0,  0, -1],   # A22: m1,m2,m4,m7(-1)
    ]
    # m3 = A11(B21-B22): only A11, not A12. Correct.
    # m7 = (A12-A22)(B21+B22): A12→1, A22→-1. Correct.
    
    V_S = [
        [1,  1,  0, -1,  0,  1,  0],   # B11: m1,m2,m4(-1),m6
        [0,  0,  0,  1,  0,  1,  0],   # B12: m4,m6
        [0,  0,  1,  0,  0,  0,  1],   # B21: m3,m7
        [1,  0, -1,  0,  1,  0,  1],   # B22: m1,m3(-1),m5,m7
    ]
    
    W_S = [
        [1,  0,  0,  1, -1,  0, -1],   # C11: m1+m4-m5-m7
        [0,  0,  1,  0,  1,  0,  0],   # C12: m3+m5
        [0,  1,  0,  1,  0,  0,  0],   # C21: m2+m4
        [1, -1, 1,  0,  0, -1,  0],   # C22: m1-m2+m3-m6
    ]
    
    # --- M_{2,2,3} rank 11 decomposition ---
    # We need a rank-11 decomposition for 2x2x3.
    # One known construction: split into two parts.
    # Actually, a known rank-11 decomposition exists. Let me use a standard one.
    # The tensor M_{2,2,3}: C = AB, A is 2x2, B is 2x3.
    # Schoolbook rank is 12. We need rank 11.
    #
    # One approach: use the fact that M_{2,2,3} can be built from M_{2,2,2} and M_{2,2,1}.
    # But rank(M_{2,2,1}) = 4 (schoolbook, since it's just 4 scalar mults).
    # That gives 7+4=11! And the decompositions can be combined (direct sum).
    #
    # So: M_{2,2,3} = M_{2,2,2} ⊕ M_{2,2,1} (in the sense of splitting B's columns)
    # B is 2x3 = [B_left(2x2) | B_right(2x1)]
    # C = [A*B_left | A*B_right]
    #
    # M_{2,2,2} part: rank 7, using Strassen as above.
    # M_{2,2,1} part: rank 4, schoolbook.
    # Total rank: 7 + 4 = 11.
    
    # M_{2,2,1} schoolbook:
    # U_1: 4x4, V_1: 2x4 (B is 2x1, so V has 2*1=2 rows), W_1: 2x4 (C is 2x1, so W has 2*1=2 rows)
    # Wait, dimensions: for M_{n,m,p}, U is n*m x rank, V is m*p x rank, W is p*n x rank.
    # For M_{2,2,1}: n=2, m=2, p=1. U: 4x4, V: 2x4, W: 2x4.
    # Schoolbook:
    U_1 = [
        [1, 0, 0, 0],   # A11
        [0, 1, 0, 0],   # A12
        [0, 0, 1, 0],   # A21
        [0, 0, 0, 1],   # A22
    ]
    V_1 = [
        [1, 0, 1, 0],   # B11 (row 0 of B, col 0)
        [0, 1, 0, 1],   # B21 (row 1 of B, col 0)
    ]
    W_1 = [
        [1, 0, 0, 0],   # C11
        [0, 1, 0, 0],   # C21
    ]
    
    # Now build M_{2,2,3} by combining M_{2,2,2} (columns 1-2 of B) and M_{2,2,1} (column 3 of B)
    # For M_{2,2,3}: n=2, m=2, p=3. U: 4x11, V: 6x11, W: 6x11.
    # V rows: (j,k) -> j*3+k, j=0,1 (rows of B), k=0,1,2 (cols of B)
    #   j=0: k=0,1,2 → indices 0,1,2
    #   j=1: k=0,1,2 → indices 3,4,5
    # W rows: (k,i) -> k*2+i, k=0,1,2 (cols of C), i=0,1 (rows of C)
    #   k=0: i=0,1 → indices 0,1
    #   k=1: i=0,1 → indices 2,3
    #   k=2: i=0,1 → indices 4,5
    #
    # M_{2,2,2} part uses B columns 0,1 (the 2x2 part):
    #   In M_{2,2,2}, B is 2x2 with rows j=0,1 and cols k=0,1.
    #   V_S rows: B11(j=0,k=0)→idx0, B12(j=0,k=1)→idx1, B21(j=1,k=0)→idx2, B22(j=1,k=1)→idx3
    #   For M_{2,2,3}, these map to: B11→V[0*3+0]=V[0], B12→V[0*3+1]=V[1], B21→V[1*3+0]=V[3], B22→V[1*3+1]=V[4]
    #   W_S rows: C11→W[0], C12→W[1], C21→W[2], C22→W[3]
    #   For M_{2,2,3}: C11→W[0*2+0]=W[0], C12→W[0*2+1]=W[1], C21→W[1*2+0]=W[2], C22→W[1*2+1]=W[3]
    #
    # M_{2,2,1} part uses B column 2 (the 2x1 part):
    #   In M_{2,2,1}, B is 2x1 with rows j=0,1 and col k=0.
    #   V_1 rows: B11→idx0, B21→idx1
    #   For M_{2,2,3}: B at (j=0,k=2)→V[0*3+2]=V[2], B at (j=1,k=2)→V[1*3+2]=V[5]
    #   W_1 rows: C11→idx0, C21→idx1
    #   For M_{2,2,3}: C at (k=2,i=0)→W[2*2+0]=W[4], C at (k=2,i=1)→W[2*2+1]=W[5]
    
    U_223 = []
    V_223 = [[0]*11 for _ in range(6)]
    W_223 = [[0]*11 for _ in range(6)]
    
    # Part 1: M_{2,2,2} in columns 0-1 (rank 7, columns 0-6 of U_223)
    for i in range(4):
        U_223.append([0]*11)
    for i in range(4):
        for c in range(7):
            U_223[i][c] = U_S[i][c]
    
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V_223[j*3+k][c] = V_S[j*2+k][c]
    
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W_223[k*2+i][c] = W_S[k*2+i][c]
    
    # Part 2: M_{2,2,1} in column 2 (rank 4, columns 7-10 of U_223)
    for i in range(4):
        for c in range(4):
            U_223[i][7+c] = U_1[i][c]
    
    for j in range(2):
        for c in range(4):
            V_223[j*3+2][7+c] = V_1[j][c]
    
    for i in range(2):
        for c in range(4):
            W_223[2*2+i][7+c] = W_1[i][c]
    
    # Now build M_{4,4,6} = M_{2,2,3} ⊗ M_{2,2,2}
    # M_{2,2,3}: n=2, m=2, p=3
    # M_{2,2,2}: n=2, m=2, p=2
    # M_{4,4,6}: n=4, m=4, p=6
    #
    # Kronecker product of tensor decompositions:
    # U_total: 16 x 77
    # V_total: 24 x 77
    # W_total: 24 x 77
    #
    # For Kronecker: 
    # U_K[(i1*2+i2) * (m1*m2) + (j1*m2+j2)][c1*7+c2] = U_223[i1*2+i2][c1] * U_S[j1*2+j2][c2]
    # Wait, let me think about indexing more carefully.
    #
    # M_{4,4,6}: A is 4x4, B is 4x6.
    # A[i,j] where i=0..3, j=0..3.
    # B[j,k] where j=0..3, k=0..5.
    # C[i,k] where i=0..3, k=0..5.
    #
    # We view A as a 2x2 block matrix with 2x2 blocks:
    #   A = [[A11, A12], [A21, A22]] where each Aab is 2x2.
    #   A[i,j] = A_{i1,i2; j1,j2} where i = i1*2+i2, j = j1*2+j2.
    #   i1 = i//2, i2 = i%2, j1 = j//2, j2 = j%2.
    #
    # Similarly B is 4x6 = 2x3 block matrix with 2x2 blocks:
    #   B = [[B11, B12, B13], [B21, B22, B23]] where each Bab is 2x2.
    #   B[j,k] = B_{j1,j2; k1,k2} where j = j1*2+j2, k = k1*2+k2.
    #   j1 = j//2, j2 = j%2, k1 = k//2, k2 = k%2.
    #
    # C is 4x6 = 2x3 block matrix with 2x2 blocks:
    #   C[i,k] = C_{i1,i2; k1,k2} where i = i1*2+i2, k = k1*2+k2.
    #
    # The Kronecker product approach:
    # C_{i1,i2; k1,k2} = sum over j1,j2 of A_{i1,i2; j1,j2} * B_{j1,j2; k1,k2}
    #
    # Using the M_{2,2,3} decomposition for the block-level multiplication:
    #   (block A)_{i1,j1} is 2x2, (block B)_{j1,k1} is 2x2.
    #   The block multiplication: C_block_{i1,k1} = sum_{j1} A_block_{i1,j1} * B_block_{j1,k1}
    #   But each term is a 2x2 matrix product.
    #
    # Actually, the Kronecker product of tensor decompositions works as follows:
    # If T1 = sum_c U1 ⊗ V1 ⊗ W1 (for M_{n1,m1,p1}) and T2 = sum_d U2 ⊗ V2 ⊗ W2 (for M_{n2,m2,p2}),
    # then T1 ⊗ T2 = sum_{c,d} (U1⊗U2) ⊗ (V1⊗V2) ⊗ (W1⊗W2) for M_{n1*n2, m1*m2, p1*p2}.
    #
    # Here n1=2,m1=2,p1=3 and n2=2,m2=2,p2=2.
    # So M_{4,4,6} = M_{2,2,3} ⊗ M_{2,2,2}.
    #
    # U_total shape: (n1*m1) * (n2*m2) = 4*4 = 16
    # V_total shape: (m1*p1) * (m2*p2) = 6*4 = 24
    # W_total shape: (p1*n1) * (p2*n2) = 6*4 = 24
    #
    # U_total[(i1*m1+j1)*(n2*m2) + (i2*m2+j2)][c1*rank2+c2] = U_223[i1*m1+j1][c1] * U_S[i2*m2+j2][c2]
    # But wait, the indexing needs to be consistent.
    #
    # For M_{n1,m1,p1}, U is indexed by (i,j) → i*m1+j, i=0..n1-1, j=0..m1-1.
    # For M_{n2,m2,p2}, U is indexed by (i',j') → i'*m2+j', i'=0..n2-1, j'=0..m2-1.
    # For the Kronecker product M_{n1*n2, m1*m2, p1*p2}, U is indexed by (I,J) → I*(m1*m2)+J.
    # I = i1*n2+i2, J = j1*m2+j2.
    # So U_total[I*(m1*m2)+J] = U_total[(i1*n2+i2)*(m1*m2) + (j1*m2+j2)]
    #   = U_223[i1*m1+j1] * U_S[i2*m2+j2]
    #
    # Let me define:
    # n1=2, m1=2, p1=3 (for M_{2,2,3})
    # n2=2, m2=2, p2=2 (for M_{2,2,2})
    # rank1=11, rank2=7
    # rank_total = 11*7 = 77
    
    n1, m1, p1 = 2, 2, 3
    n2, m2, p2 = 2, 2, 2
    rank1, rank2 = 11, 7
    rank_total = rank1 * rank2
    
    U_total = [[0]*rank_total for _ in range(n1*n2*m1*m2)]
    V_total = [[0]*rank_total for _ in range(m1*p1*m2*p2)]
    W_total = [[0]*rank_total for _ in range(p1*n1*p2*n2)]
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1*rank2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U_223[i1*m1+j1][c1]
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U_S[i2*m2+j2][c2]
                            I = i1*n2 + i2
                            J = j1*m2 + j2
                            U_total[I*(m1*m2)+J][c] = u1_val * u2_val
            
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V_223[j1*p1+k1][c1]
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V_S[j2*p2+k2][c2]
                            J = j1*m2 + j2
                            K = k1*p2 + k2
                            V_total[J*(p1*p2)+K][c] = v1_val * v2_val
            
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W_223[k1*n1+i1][c1]
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W_S[k2*n2+i2][c2]
                            K = k1*p2 + k2
                            I = i1*n2 + i2
                            W_total[K*(n1*n2)+I][c] = w1_val * w2_val
    
    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
