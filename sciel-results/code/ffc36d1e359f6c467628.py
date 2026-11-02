def construct():
    n, m, p = 2, 2, 12
    rank = 7
    U = [[0]*rank for _ in range(n*m)]  # 4 x 7
    V = [[0]*rank for _ in range(m*p)]  # 24 x 7
    W = [[0]*rank for _ in range(p*n)]  # 24 x 7
    
    # Strassen's 7 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # A indexing: A[i,j] at position i*m+j = i*2+j
    # A00=0, A01=1, A10=2, A11=3
    # B indexing: B[j,k] at position j*p+k = j*12+k
    # B00: j=0,k=0 -> pos 0; B01: j=0,k=1 -> pos 1; ...
    # B row 0: positions 0..11; B row 1: positions 12..23
    # C indexing: C[i,k] at position k*n+i = k*2+i
    # C00: i=0,k=0 -> pos 0; C01: i=1,k=0 -> pos 1; C10: i=0,k=1 -> pos 2; C11: i=1,k=1 -> pos 3
    # Wait, let me re-read: W[k*n+i, r] so row index = k*n + i = k*2 + i
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    for k in range(p):
        V[k][0] = 1      # B[0,k] = B00 part
        V[p+k][0] = 1    # B[1,k] = B11 part
    for k in range(p):
        W[k*n+0][0] = 1  # C[0,k] = C00
        W[k*n+1][0] = 1  # C[1,k] = C11
    
    # Term 1: M2 = (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    for k in range(p):
        V[k][1] = 1      # B[0,k]
    for k in range(p):
        W[k*n+1][1] = 1  # C[1,k] = C10
    
    # Term 2: M3 = A00(B01-B11)
    U[0][2] = 1  # A00
    for k in range(p):
        V[k][2] = 1      # B[0,k]
        V[p+k][2] = -1   # B[1,k]
    for k in range(p):
        W[k*n+1][2] = 1  # C[0,k] = C01
        W[k*n+3][2] = 1  # C[1,k] = C11
    
    # Term 3: M4 = A11(B10-B00)
    U[3][3] = 1  # A11
    for k in range(p):
        V[p+k][3] = 1   # B[1,k]
        V[k][3] = -1    # B[0,k]
    for k in range(p):
        W[k*n+0][3] = 1  # C[0,k] = C00
        W[k*n+1][3] = 1  # C[1,k] = C10
    
    # Term 4: M5 = (A00+A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    for k in range(p):
        V[p+k][4] = 1   # B[1,k]
    for k in range(p):
        W[k*n+0][4] = -1  # C[0,k] = C00 (negative)
        W[k*n+1][4] = 1   # C[0,k] = C01 (positive)
    
    # Term 5: M6 = (A10-A00)(B00+B10)
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    for k in range(p):
        V[k][5] = 1      # B[0,k]
        V[p+k][5] = 1    # B[1,k]
    for k in range(p):
        W[k*n+3][5] = 1  # C[1,k] = C11
    
    # Term 6: M7 = (A01-A11)(B01+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    for k in range(p):
        V[k][6] = 1      # B[0,k]
        V[p+k][6] = 1    # B[1,k]
    for k in range(p):
        W[k*n+0][6] = 1  # C[0,k] = C00
        W[k*n+3][6] = 1  # C[1,k] = C11
    
    return {"U": U, "V": V, "W": W}
