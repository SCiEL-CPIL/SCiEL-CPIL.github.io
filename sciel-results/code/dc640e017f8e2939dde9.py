def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # A is 2x2, B is 2x2, C = A@B is 2x2
    # Indexing: A[i*m+j] with m=2, so A[0]=a00, A[1]=a01, A[2]=a10, A[3]=a11
    # B[j*p+k] with p=2, so B[0]=b00, B[1]=b01, B[2]=b10, B[3]=b11
    # C[k*n+i] with n=2, so C[0]=c00, C[1]=c01, C[2]=c10, C[3]=c11
    
    # Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    U = [[1,1,0,0, 1,0,0,0, 0,0,1,1, 1,0,0,0, 0,0,1,0, 1,1,0,0, 0,1,1,0]]
    V = [[1,0,1,0, 1,0,0,0, 0,1,0,1, 0,1,0,1, 0,0,1,1, 1,1,0,0, 0,0,1,1]]
    W = [[1,0,0,1, 0,0,0,0, 0,1,0,0, 1,0,0,0, 0,0,1,0, -1,0,0,0, 0,1,0,0, 0,0,0,1]]
    # Wait, W needs to be (p*n, rank) = (4, 7)
    # C[k*n+i, r] where n=2
    # c00 is C[0], c01 is C[1], c10 is C[2], c11 is C[3]
    
    # Let me redo this more carefully
    # U has shape (n*m, rank) = (4, 7)
    # V has shape (m*p, rank) = (4, 7)  
    # W has shape (p*n, rank) = (4, 7)
    
    # U[i*m+j, r] = coefficient of A[i,j] in the r-th bilinear form
    # V[j*p+k, r] = coefficient of B[j,k] in the r-th bilinear form
    # W[k*n+i, r] = coefficient of C[i,k] that the r-th bilinear form contributes to
    
    # M1 = (a00+a11)(b00+b11)
    # U: a00=U[0], a11=U[3] → U[0,r0]=1, U[3,r0]=1
    # V: b00=V[0], b11=V[3] → V[0,r0]=1, V[3,r0]=1
    # W: contributes to c00=W[0], c11=W[3] → W[0,r0]=1, W[3,r0]=1
    
    # Let me build this properly
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a00+a11)(b00+b11) → c00 += M1, c11 += M1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (a10+a11)(b00) → c10 += M2, c11 -= M2
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    # M3 = a00(b01-b11) → c01 += M3, c11 += M3
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # M4 = a11(b10-b00) → c00 += M4, c10 += M4
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # M5 = (a00+a01)(b11) → c00 -= M5, c01 += M5
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (a10-a00)(b00+b01) → c11 += M6
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (a01-a11)(b10+b11) → c00 += M7
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W
