    # Block 4: A12 (2x3) x B22 (3x3) -> C12 (2x3)
    # Strassen for A12[:,0:2] x B22[0:2,0:2]
    col = strassen(0, 2, 2, U, V, W, col)
    # Schoolbook for A12[:,0:2] x B22[0:2,2]
    U[0*m+2][col] = 1; V[2*p+4][col] = 1; W[4*n+0][col] = 1; col += 1
    U[0*m+3][col] = 1; V[3*p+4][col] = 1; W[4*n+0][col] = 1; col += 1
    U[1*m+2][col] = 1; V[2*p+4][col] = 1; W[4*n+1][col] = 1; col += 1
    U[1*m+3][col] = 1; V[3*p+4][col] = 1; W[4*n+1][col] = 1; col += 1
    # Schoolbook for A12[:,2] x B22[2,:]
    U[0*m+4][col] = 1; V[4*p+2][col] = 1; W[2*n+0][col] = 1; col += 1
    U[0*m+4][col] = 1; V[4*p+3][col] = 1; W[3*n+0][col] = 1; col += 1
    U[0*m+4][col] = 1; V[4*p+4][col] = 1; W[4*n+0][col] = 1; col += 1
    U[1*m+4][col] = 1; V[4*p+2][col] = 1; W[2*n+1][col] = 1; col += 1
    U[1*m+4][col] = 1; V[4*p+3][col] = 1; W[3*n+1][col] = 1; col += 1
    U[1*m+4][col] = 1; V[4*p+4][col] = 1; W[4*n+1][col] = 1; col += 1
