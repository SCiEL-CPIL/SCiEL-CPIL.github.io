"""Matrix multiplication 4x4 * 4x9 with Strassen-based block decomposition."""
DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    # Split p=9 into two 4-blocks and one 1-block: 4+4+1
    # Subproblem 1: 4x4 * 4x4 -> rank 49 (Strassen)
    # Subproblem 2: 4x4 * 4x4 -> rank 49 (Strassen)
    # Subproblem 3: 4x4 * 4x1 -> rank 16 (schoolbook)
    # Total rank = 49 + 49 + 16 = 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 4x4x4 decomposition for first 4x4 block (columns 0-3 of B)
    # Strassen for 2x2x2 has rank 7, tensor product gives 4x4x4 with rank 49
    # M = A*B where A is 4x4, B is 4x4
    # Using 2x2 block structure: A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # Strassen: 
    #   P1 = (A11+A22)(B11+B22)
    #   P2 = (A21+A22)B11
    #   P3 = A11(B12-B22)
    #   P4 = A22(B21-B11)
    #   P5 = (A11+A12)B22
    #   P6 = (A21-A11)(B11+B12)
    #   P7 = (A12-A22)(B21+B22)
    #   C11 = P1+P4-P5+P7
    #   C12 = P3+P5
    #   C21 = P2+P4
    #   C22 = P1-P2+P3+P6
    
    # For 4x4x4 via tensor product of two 2x2x2 Strassen:
    # Each 2x2x2 Strassen has 7 terms, so 4x4x4 has 7*7=49 terms
    # The indices: for the 4x4 matrix, row index = 2*a + a', col index = 2*b + b'
    # where a,a',b,b' in {0,1}
    
    # Strassen 2x2x2: U[a*2+b, c], V[b*2+d, c], W[d*2+a, c] for c in 0..6
    strassen_2x2 = [
        # (a,b) -> coefficient for each of 7 products
        # P1 = (A00+A11)(B00+B11): A00*B00, A00*B11, A11*B00, A11*B11
        # P2 = (A10+A11)B00: A10*B00, A11*B00
        # P3 = A00(B01-B11): A00*B01, -A00*B11
        # P4 = A11(B10-B00): A11*B10, -A11*B00
        # P5 = (A00+A01)B11: A00*B11, A01*B11
        # P6 = (A10-A00)(B00+B01): A10*B00, A10*B01, -A00*B00, -A00*B01
        # P7 = (A01-A11)(B10+B11): A01*B10, A01*B11, -A11*B10, -A11*B11
    ]
    
    # Let me directly construct the 4x4x4 Strassen decomposition
    # For 4x4 matrices indexed by (r,c) where r=2*i+i2, c=2*j+j2, i,j,i2,j2 in {0,1}
    # The 49 products correspond to pairs (s1, s2) where s1,s2 in {0..6}
    
    # Strassen 2x2x2 factor matrices (rank 7):
    # U2[i*2+j, c] for i,j in {0,1}, c in {0..6}
    # V2[j*2+k, c] for j,k in {0,1}, c in {0..6}
    # W2[k*2+i, c] for k,i in {0,1}, c in {0..6}
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 0, 0, 0],   # A01 (not used directly, appears in P5,P7)
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, 0],   # A11
    ]
    # Wait, let me be more careful. Let me use the standard Strassen formulation.
    
    # For 2x2x2: C = A*B, A,B,C are 2x2
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # P1 = (A00+A11)(B00+B11): contributes to C00 and C11
    #   U: A00:1, A11:1 -> U[0,0]=1, U[3,0]=1
    #   V: B00:1, B11:1 -> V[0,0]=1, V[3,0]=1
    #   W: C00:1, C11:1 -> W[0,0]=1, W[3,0]=1
    # P2 = (A10+A11)B00: contributes to C10
    #   U: A10:1, A11:1 -> U[2,1]=1, U[3,1]=1
    #   V: B00:1 -> V[0,1]=1
    #   W: C10:1 -> W[2,1]=1  (k=0,i=1 -> k*2+i=2)
    # P3 = A00(B01-B11): contributes to C01
    #   U: A00:1 -> U[0,2]=1
    #   V: B01:1, B11:-1 -> V[1,2]=1, V[3,2]=-1
    #   W: C01:1 -> W[1,2]=1  (k=1,i=0 -> k*2+i=1)
    # P4 = A11(B10-B00): contributes to C10
    #   U: A11:1 -> U[3,3]=1
    #   V: B10:1, B00:-1 -> V[2,3]=1, V[0,3]=-1
    #   W: C10:1 -> W[2,3]=1
    # P5 = (A00+A01)B11: contributes to C01
    #   U: A00:1, A01:1 -> U[0,4]=1, U[1,4]=1
    #   V: B11:1 -> V[3,4]=1
    #   W: C01:1 -> W[1,4]=1
    # P6 = (A10-A00)(B00+B01): contributes to C10
    #   U: A10:1, A00:-1 -> U[2,5]=1, U[0,5]=-1
    #   V: B00:1, B01:1 -> V[0,5]=1, V[1,5]=1
    #   W: C10:1 -> W[2,5]=1
    # P7 = (A01-A11)(B10+B11): contributes to C11
    #   U: A01:1, A11:-1 -> U[1,6]=1, U[3,6]=-1
    #   V: B10:1, B11:1 -> V[2,6]=1, V[3,6]=1
    #   W: C11:1 -> W[3,7]=1... wait W[3,6]=1 (k=1,i=1 -> k*2+i=3)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # P1: c=0
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    # P2: c=1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    # P3: c=2
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    # P4: c=3
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1
    # P5: c=4
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[1][4] = 1
    # P6: c=5
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[2][5] = 1
    # P7: c=6
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[3][6] = 1
    
    # Now for 4x4x4: use tensor product of two 2x2x2 Strassen
    # A is 4x4, index (i,j) where i = 2*i1 + i2, j = 2*j1 + j2
    # B is 4x4, index (j,k) where j = 2*j1 + j2, k = 2*k1 + k2
    # C is 4x4, index (i,k) where i = 2*i1 + i2, k = 2*k1 + k2
    
    # The 49 terms are indexed by (c1, c2) where c1,c2 in {0..6}
    # For term (c1, c2):
    #   U[(2*i1+i2)*4 + (2*j1+j2), c1*7+c2] = U2[i1*2+i2, c1] * U2[j1*2+j2, c2]
    #   V[(2*j1+j2)*4 + (2*k1+k2), c1*7+c2] = V2[j1*2+j2, c1] * V2[k1*2+k2, c2]
    #   W[(2*k1+k2)*4 + (2*i1+i2), c1*7+c2] = W2[k1*2+k2, c1] * W2[i1*2+i2, c2]
    
    def add_strassen_4x4x4(U, V, W, col_offset, row_offset_U=0, row_offset_V=0, row_offset_W=0):
        """Add a 4x4x4 Strassen decomposition starting at col_offset.
        U rows are offset by row_offset_U, V rows by row_offset_V, W rows by row_offset_W."""
        for c1 in range(7):
            for c2 in range(7):
                c = col_offset + c1 * 7 + c2
                for i1 in range(2):
                    for i2 in range(2):
                        for j1 in range(2):
                            for j2 in range(2):
                                for k1 in range(2):
                                    for k2 in range(2):
                                        u_val = U2[i1*2+i2][c1] * U2[j1*2+j2][c2]
                                        v_val = V2[j1*2+j2][c1] * V2[k1*2+k2][c2]
                                        w_val = W2[k1*2+k2][c1] * W2[i1*2+i2][c2]
                                        if u_val != 0:
                                            U[row_offset_U + (2*i1+i2)*4 + (2*j1+j2)][c] += u_val
                                        if v_val != 0:
                                            V[row_offset_V + (2*j1+j2)*4 + (2*k1+k2)][c] += v_val
                                        if w_val != 0:
                                            W[row_offset_W + (2*k1+k2)*4 + (2*i1+i2)][c] += w_val
    
    # Subproblem 1: 4x4 * 4x4 (B columns 0-3), result in C columns 0-3
    # A is full 4x4, B is 4x4 (cols 0-3), C is 4x4 (cols 0-3)
    add_strassen_4x4x4(U, V, W, col_offset=0, row_offset_U=0, row_offset_V=0, row_offset_W=0)
    
    # Subproblem 2: 4x4 * 4x4 (B columns 4-7), result in C columns 4-7
    add_strassen_4x4x4(U, V, W, col_offset=49, row_offset_U=0, row_offset_V=16, row_offset_W=16)
    
    # Subproblem 3: 4x4 * 4x1 (B column 8), result in C column 8
    # Schoolbook: rank 16
    # U[i*4+j, c] = 1, V[j*1+0, c] = 1, W[0*4+i, c] = 1 for each (i,j)
    for i in range(4):
        for j in range(4):
            c = 98 + i*4 + j
            U[i*4+j][c] = 1
            V[j*1+0][c] = 1  # j*1+0 = j
            W[0*4+i][c] = 1  # 0*4+i = i
    
    return {"U": U, "V": V, "W": W}
