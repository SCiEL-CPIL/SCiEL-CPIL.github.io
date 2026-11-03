def _strassen_44():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 12 x 44
    V = [[0]*rank for _ in range(m*p)]  # 16 x 44
    W = [[0]*rank for _ in range(p*n)]  # 12 x 44
    
    col = 0
    
    # Strassen 2x2x2x2 for blocks (i in {0,1}, j in {j0,j0+1}, k in {k0,k0+1})
    def add_strassen(i0, j0, k0):
        nonlocal col
        # i0, j0, k0 are the starting indices
        # A block: rows i0..i0+1, cols j0..j0+1 (2x2)
        # B block: rows j0..j0+1, cols k0..k0+1 (2x2)
        # C block: rows i0..i0+1, cols k0..k0+1 (2x2)
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11 at (i0,j0), A22 at (i0+1,j0+1)
        # V: B11 at (j0,k0), B22 at (j0+1,k0+1)
        # W: C11 at (k0,i0), C22 at (k0+1,i0+1)
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M3 = A11(B21-B22)
        U[i0*m + j0][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = -1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M4 = A22(B12-B11)
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + (k0+1)][col] = 1
        V[j0*p + k0][col] = -1
        W[k0*n + (i0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[k0*n + (i0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
    
    # Schoolbook 1x2x2x2 for block (i=2, j in {j0,j0+1}, k in {k0,k0+1})
    def add_schoolbook_1x2x2x2(i, j0, k0):
        nonlocal col
        for jj in range(2):
            for kk in range(2):
                U[i*m + j0 + jj][col] = 1
                V[(j0+jj)*p + k0 + kk][col] = 1
                W[(k0+kk)*n + i][col] = 1
                col += 1
    
    # Four Strassen blocks for i in {0,1}
    add_strassen(0, 0, 0)  # i=0,1; j=0,1; k=0,1
    add_strassen(0, 2, 0)  # i=0,1; j=2,3; k=0,1
    add_strassen(0, 0, 2)  # i=0,1; j=0,1; k=2,3
    add_strassen(0, 2, 2)  # i=0,1; j=2,3; k=2,3
    
    # Four schoolbook blocks for i=2
    add_schoolbook_1x2x2x2(2, 0, 0)  # i=2; j=0,1; k=0,1
    add_schoolbook_1x2x2x2(2, 2, 0)  # i=2; j=2,3; k=0,1
    add_schoolbook_1x2x2x2(2, 0, 2)  # i=2; j=0,1; k=2,3
    add_schoolbook_1x2x2x2(2, 2, 2)  # i=2; j=2,3; k=2,3
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
