"""Optimized matrix multiplication 2x5 * 5x5 using Strassen on 2x2 blocks."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    # Partition: 2+2+1 along the m and p dimensions
    # A is 2x5: [A1(2x2), A2(2x2), A3(2x1)]
    # B is 5x5: [[B11(2x2), B12(2x2), B13(2x1)],
    #            [B21(2x2), B22(2x2), B23(2x1)],
    #            [B31(1x2), B32(1x2), B33(1x1)]]
    # C is 2x5: [C1(2x2), C2(2x2), C3(2x1)]
    # C1 = A1*B11 + A2*B21 + A3*B31
    # C2 = A1*B12 + A2*B22 + A3*B32
    # C3 = A1*B13 + A2*B23 + A3*B33
    
    # We'll build the decomposition column by column.
    # Each column corresponds to one scalar multiplication in the bilinear algorithm.
    # U has shape (n*m, rank) = (10, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (10, rank)
    
    # Index conventions:
    # A[i,j] -> U index i*m+j = i*5+j
    # B[j,k] -> V index j*p+k = j*5+k
    # C[i,k] -> W index k*n+i = k*2+i
    
    cols_U = []  # list of (index, coeff) for U
    cols_V = []
    cols_W = []
    
    def add_col(u_entries, v_entries, w_entries):
        """Add one column to the decomposition."""
        cols_U.append(u_entries)
        cols_V.append(v_entries)
        cols_W.append(w_entries)
    
    # Strassen for 2x2 * 2x2:
    # Given matrices P(2x2) and Q(2x2), compute R = P*Q
    # Using Strassen:
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q01-Q11)
    # M4 = P11*(Q10-Q00)
    # M5 = (P00+P01)*Q11
    # M6 = (P10-P00)*(Q00+Q01)
    # M7 = (P01-P11)*(Q10+Q11)
    # R00 = M1+M4-M5+M7
    # R01 = M3+M5
    # R10 = M2+M4
    # R11 = M1-M2+M3+M6
    
    # For A1(2x2) * B11(2x2) contributing to C1(2x2):
    # A1 = [[a00, a01], [a10, a11]] (A indices: 0,1,5,6)
    # B11 = [[b00, b01], [b10, b11]] (B indices: 0,1,5,6)
    # C1 = [[c00, c01], [c10, c11]] (C indices: 0,1,2,3)
    
    # Strassen for A1*B11:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C1_00 = M1+M4-M5+M7
    # C1_01 = M3+M5
    # C1_10 = M2+M4
    # C1_11 = M1-M2+M3+M6
    
    # A1 indices: a00=0, a01=1, a10=5, a11=6
    # B11 indices: b00=0, b01=1, b10=5, b11=6
    # C1 indices: c00=0, c01=1, c10=2, c11=3
    
    # M1
    add_col([(0,1),(6,1)], [(0,1),(6,1)], [(0,1)])
    # M2
    add_col([(5,1),(6,1)], [(0,1)], [(2,1)])
    # M3
    add_col([(0,1)], [(1,1),(6,-1)], [(1,1)])
    # M4
    add_col([(6,1)], [(5,1),(0,-1)], [(0,1),(2,1)])
    # M5
    add_col([(0,1),(1,1)], [(6,1)], [(1,1),(0,-1)])
    # M6
    add_col([(5,1),(0,-1)], [(0,1),(1,1)], [(3,1)])
    # M7
    add_col([(1,1),(6,-1)], [(5,1),(6,1)], [(0,1),(3,-1)])
    
    # Strassen for A2*B22 contributing to C2(2x2):
    # A2 = [[a20, a21], [a30, a31]] (A indices: 2,3,7,8)
    # B22 = [[b20, b21], [b30, b31]] (B indices: 10,11,15,16)
    # C2 = [[c20, c21], [c30, c31]] (C indices: 4,5,6,7)
    
    # M1' = (a20+a31)*(b20+b31)
    # M2' = (a30+a31)*b20
    # M3' = a20*(b21-b31)
    # M4' = a31*(b30-b20)
    # M5' = (a20+a21)*b31
    # M6' = (a30-a20)*(b20+b21)
    # M7' = (a21-a31)*(b30+b31)
    # C2_00 = M1'+M4'-M5'+M7'
    # C2_01 = M3'+M5'
    # C2_10 = M2'+M4'
    # C2_11 = M1'-M2'+M3'+M6'
    
    # M1'
    add_col([(2,1),(8,1)], [(10,1),(16,1)], [(4,1)])
    # M2'
    add_col([(7,1),(8,1)], [(10,1)], [(6,1)])
    # M3'
    add_col([(2,1)], [(11,1),(16,-1)], [(5,1)])
    # M4'
    add_col([(8,1)], [(15,1),(10,-1)], [(4,1),(6,1)])
    # M5'
    add_col([(2,1),(3,1)], [(16,1)], [(5,1),(4,-1)])
    # M6'
    add_col([(7,1),(2,-1)], [(10,1),(11,1)], [(7,1)])
    # M7'
    add_col([(3,1),(8,-1)], [(15,1),(16,1)], [(4,1),(7,-1)])
    
    # Strassen for A1*B12 contributing to C2(2x2):
    # A1 = [[a00, a01], [a10, a11]] (A indices: 0,1,5,6)
    # B12 = [[b02, b03], [b12, b13]] (B indices: 2,3,7,8)
    # C2 contribution: [[c20, c21], [c30, c31]] (C indices: 4,5,6,7)
    
    # M1'' = (a00+a11)*(b02+b13)
    # M2'' = (a10+a11)*b02
    # M3'' = a00*(b03-b13)
    # M4'' = a11*(b12-b02)
    # M5'' = (a00+a01)*b13
    # M6'' = (a10-a00)*(b02+b03)
    # M7'' = (a01-a11)*(b12+b13)
    # C2_00 += M1''+M4''-M5''+M7''
    # C2_01 += M3''+M5''
    # C2_10 += M2''+M4''
    # C2_11 += M1''-M2''+M3''+M6''
    
    # M1''
    add_col([(0,1),(6,1)], [(2,1),(8,1)], [(4,1)])
    # M2''
    add_col([(5,1),(6,1)], [(2,1)], [(6,1)])
    # M3''
    add_col([(0,1)], [(3,1),(8,-1)], [(5,1)])
    # M4''
    add_col([(6,1)], [(7,1),(2,-1)], [(4,1),(6,1)])
    # M5''
    add_col([(0,1),(1,1)], [(8,1)], [(5,1),(4,-1)])
    # M6''
    add_col([(5,1),(0,-1)], [(2,1),(3,1)], [(7,1)])
    # M7''
    add_col([(1,1),(6,-1)], [(7,1),(8,1)], [(4,1),(7,-1)])
    
    # Strassen for A2*B21 contributing to C1(2x2):
    # A2 = [[a20, a21], [a30, a31]] (A indices: 2,3,7,8)
    # B21 = [[b20, b21], [b30, b31]] (B indices: 10,11,15,16)
    # Wait, B21 is rows 2-3, cols 0-1 of B. B indices: row j, col k -> j*5+k
    # B21 = [[b20, b21], [b30, b31]] = B[2,0],B[2,1],B[3,0],B[3,1] = indices 10,11,15,16
    
    # Hmm wait, I already used B21 for the Strassen block above. Let me recheck.
    # B is 5x5. B[j,k] has index j*5+k.
    # B11 = B[0:2,0:2] = indices 0,1,5,6
    # B12 = B[0:2,2:4] = indices 2,3,7,8
    # B13 = B[0:2,4:5] = indices 4,9
    # B21 = B[2:4,0:2] = indices 10,11,15,16
    # B22 = B[2:4,2:4] = indices 12,13,17,18
    # B23 = B[2:4,4:5] = indices 14,19
    # B31 = B[4:5,0:2] = indices 20,21
    # B32 = B[4:5,2:4] = indices 22,23
    # B33 = B[4:5,4:5] = index 24
    
    # Oh! I made an error. B22 should be indices 12,13,17,18 not 10,11,15,16.
    # Let me redo this.
    
    # Let me restart with correct indices.
    cols_U = []
    cols_V = []
    cols_W = []
    
    # A is 2x5: A[i,j] -> index i*5+j
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3, A[0,4]=4
    # A[1,0]=5, A[1,1]=6, A[1,2]=7, A[1,3]=8, A[1,4]=9
    
    # B is 5x5: B[j,k] -> index j*5+k
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[0,4]=4
    # B[1,0]=5, B[1,1]=6, B[1,2]=7, B[1,3]=8, B[1,4]=9
    # B[2,0]=10, B[2,1]=11, B[2,2]=12, B[2,3]=13, B[2,4]=14
    # B[3,0]=15, B[3,1]=16, B[3,2]=17, B[3,3]=18, B[3,4]=19
    # B[4,0]=20, B[4,1]=21, B[4,2]=22, B[4,3]=23, B[4,4]=24
    
    # C is 2x5: C[i,k] -> W index k*2+i
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[0,4]=4
    # C[1,0]=5, C[1,1]=6, C[1,2]=7, C[1,3]=8, C[1,4]=9
    
    # Partition:
    # A1 = A[:,0:2] = [[A00,A01],[A10,A11]] = indices 0,1,5,6
    # A2 = A[:,2:4] = [[A02,A03],[A12,A13]] = indices 2,3,7,8
    # A3 = A[:,4:5] = [[A04],[A14]] = indices 4,9
    
    # B11 = B[0:2,0:2] = indices 0,1,5,6
    # B12 = B[0:2,2:4] = indices 2,3,7,8
    # B13 = B[0:2,4:5] = indices 4,9
    # B21 = B[2:4,0:2] = indices 10,11,15,16
    # B22 = B[2:4,2:4] = indices 12,13,17,18
    # B23 = B[2:4,4:5] = indices 14,19
    # B31 = B[4:5,0:2] = indices 20,21
    # B32 = B[4:5,2:4] = indices 22,23
    # B33 = B[4:5,4:5] = index 24
    
    # C1 = C[:,0:2] = [[C00,C01],[C10,C11]] = W indices 0,1,5,6
    # C2 = C[:,2:4] = [[C02,C03],[C12,C13]] = W indices 2,3,7,8
    # C3 = C[:,4:5] = [[C04],[C14]] = W indices 4,9
    
    def add_col(u_entries, v_entries, w_entries):
        cols_U.append(u_entries)
        cols_V.append(v_entries)
        cols_W.append(w_entries)
    
    # Helper: Strassen for P(2x2)*Q(2x2) where P has A-indices p00,p01,p10,p11
    # and Q has B-indices q00,q01,q10,q11, contributing to C-indices c00,c01,c10,c11
    def strassen_block(p00, p01, p10, p11, q00, q01, q10, q11, c00, c01, c10, c11):
        # M1 = (p00+p11)*(q00+q11)
        add_col([(p00,1),(p11,1)], [(q00,1),(q11,1)], [(c00,1)])
        # M2 = (p10+p11)*q00
        add_col([(p10,1),(p11,1)], [(q00,1)], [(c10,1)])
        # M3 = p00*(q01-q11)
        add_col([(p00,1)], [(q01,1),(q11,-1)], [(c01,1)])
        # M4 = p11*(q10-q00)
        add_col([(p11,1)], [(q10,1),(q00,-1)], [(c00,1),(c10,1)])
        # M5 = (p00+p01)*q11
        add_col([(p00,1),(p01,1)], [(q11,1)], [(c01,1),(c00,-1)])
        # M6 = (p10-p00)*(q00+q01)
        add_col([(p10,1),(p00,-1)], [(q00,1),(q01,1)], [(c11,1)])
        # M7 = (p01-p11)*(q10+q11)
        add_col([(p01,1),(p11,-1)], [(q10,1),(q11,1)], [(c00,1),(c11,-1)])
    
    # A1*B11 -> C1
    strassen_block(0,1,5,6, 0,1,5,6, 0,1,5,6)
    
    # A2*B22 -> C2
    strassen_block(2,3,7,8, 12,13,17,18, 2,3,7,8)
    
    # A1*B12 -> C2
    strassen_block(0,1,5,6, 2,3,7,8, 2,3,7,8)
    
    # A2*B21 -> C1
    strassen_block(2,3,7,8, 10,11,15,16, 0,1,5,6)
    
    # Now the remaining terms (schoolbook):
    # A3*B31 -> C1: A3=[[A04],[A14]], B31=[[B40,B41]]
    # C1[0,0] += A04*B40, C1[0,1] += A04*B41
    # C1[1,0] += A14*B40, C1[1,1] += A14*B41
    add_col([(4,1)], [(20,1)], [(0,1)])
    add_col([(4,1)], [(21,1)], [(1,1)])
    add_col([(9,1)], [(20,1)], [(5,1)])
    add_col([(9,1)], [(21,1)], [(6,1)])
    
    # A3*B32 -> C2: A3=[[A04],[A14]], B32=[[B42,B43]]
    # C2[0,0] += A04*B42, C2[0,1] += A04*B43
    # C2[1,0] += A14*B42, C2[1,1] += A14*B43
    add_col([(4,1)], [(22,1)], [(2,1)])
    add_col([(4,1)], [(23,1)], [(3,1)])
    add_col([(9,1)], [(22,1)], [(7,1)])
    add_col([(9,1)], [(23,1)], [(8,1)])
    
    # A1*B13 -> C3: A1=[[A00,A01],[A10,A11]], B13=[[B04],[B14]]
    # C3[0] += A00*B04 + A01*B14
    # C3[1] += A10*B04 + A11*B14
    add_col([(0,1)], [(4,1)], [(4,1)])
    add_col([(1,1)], [(9,1)], [(4,1)])
    add_col([(5,1)], [(4,1)], [(9,1)])
    add_col([(6,1)], [(9,1)], [(9,1)])
    
    # A2*B23 -> C3: A2=[[A02,A03],[A12,A13]], B23=[[B24],[B34]]
    # C3[0] += A02*B24 + A03*B34
    # C3[1] += A12*B24 + A13*B34
    add_col([(2,1)], [(14,1)], [(4,1)])
    add_col([(3,1)], [(19,1)], [(4,1)])
    add_col([(7,1)], [(14,1)], [(9,1)])
    add_col([(8,1)], [(19,1)], [(9,1)])
    
    # A3*B33 -> C3: A3=[[A04],[A14]], B33=[B44]
    # C3[0] += A04*B44
    # C3[1] += A14*B44
    add_col([(4,1)], [(24,1)], [(4,1)])
    add_col([(9,1)], [(24,1)], [(9,1)])
    
    rank = len(cols_U)
    # rank should be 4*7 + 4+4+4+4+2 = 28+18 = 46
    
    # Build the matrices
    U = [[0]*rank for _ in range(10)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(10)]
    
    for c in range(rank):
        for idx, coeff in cols_U[c]:
            U[idx][c] = coeff
        for idx, coeff in cols_V[c]:
            V[idx][c] = coeff
        for idx, coeff in cols_W[c]:
            W[idx][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
