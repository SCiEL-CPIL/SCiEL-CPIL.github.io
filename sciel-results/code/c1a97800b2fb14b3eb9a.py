def strassen_2x2_2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Index convention: U[i*2+k, r], V[k*2+j, r], W[j*2+i, r]
    """
    # Strassen's 7 multiplications
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: M1 = (A00+A11)*(B00+B11)
    # A part: A[0,0] + A[1,1] → u[0]=1, u[3]=1
    # B part: B[0,0] + B[1,1] → v[0]=1, v[3]=1
    # C part: contributes to C[0,0] and C[1,1] with coeff +1
    # W index: j*2+i, so C[0,0]→w[0], C[1,1]→w[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2 = (A10+A11)*B00
    # A part: A[1,0] + A[1,1] → u[2]=1, u[3]=1
    # B part: B[0,0] → v[0]=1
    # C part: contributes to C[1,0] with coeff +1
    # C[1,0] → w[0*2+1] = w[1]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # Term 2: M3 = A00*(B01-B11)
    # A part: A[0,0] → u[0]=1
    # B part: B[0,1] - B[1,1] → v[1]=1, v[3]=-1
    # C part: contributes to C[0,1] with coeff +1
    # C[0,1] → w[1*2+0] = w[2]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # Term 3: M4 = A11*(B10-B00)
    # A part: A[1,1] → u[3]=1
    # B part: B[1,0] - B[0,0] → v[2]=1, v[0]=-1
    # C part: contributes to C[0,0] with +1 and C[1,0] with +1
    # C[0,0] → w[0], C[1,0] → w[1]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 4: M5 = (A00+A01)*B11
    # A part: A[0,0] + A[0,1] → u[0]=1, u[1]=1
    # B part: B[1,1] → v[3]=1
    # C part: contributes to C[0,0] with -1 and C[0,1] with +1
    # C[0,0] → w[0], C[0,1] → w[2]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 5: M6 = (A10-A00)*(B00+B01)
    # A part: A[1,0] - A[0,0] → u[2]=1, u[0]=-1
    # B part: B[0,0] + B[0,1] → v[0]=1, v[1]=1
    # C part: contributes to C[1,1] with +1
    # C[1,1] → w[3]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: M7 = (A01-A11)*(B10+B11)
    # A part: A[0,1] - A[1,1] → u[1]=1, u[3]=-1
    # B part: B[1,0] + B[1,1] → v[2]=1, v[3]=1
    # C part: contributes to C[0,0] with +1
    # C[0,0] → w[0]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W
