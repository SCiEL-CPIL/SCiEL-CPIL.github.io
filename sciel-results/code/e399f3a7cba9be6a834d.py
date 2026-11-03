"""Matrix multiplication 3x4x5 tensor decomposition."""
DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = 3, 4, 5
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        """Add Strassen 2x2x2 block at (i0, j0, k0)."""
        nonlocal col
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # M3 = A[i0,j0] * (B[j0+1,k0] - B[j0+1,k0+1])
        U[i0*m + j0][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = -1
        W[k0*n + (i0+1)][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M4 = A[i0+1,j0+1] * (B[j0,k0+1] - B[j0+1,k0])
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + (k0+1)][col] = 1
        V[(j0+1)*p + k0][col] = -1
        W[(k0+1)*n + i0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = -1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0+1]+B[j0,k0+1])
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[(j0+1)*p + (k0+1)][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
    
    # Four Strassen blocks
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 2)
    
    # Schoolbook for remaining terms
    # i=2, all j, k in {0,1,2,3}
    for j in range(4):
        for k in range(4):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    # i in {0,1}, all j, k=4
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
            col += 1
    
    # i=2, all j, k=4
    for j in range(4):
        U[2*m + j][col] = 1
        V[j*p + 4][col] = 1
        W[4*n + 2][col] = 1
        col += 1
    
    assert col == rank, f"col={col} != rank={rank}"
    return {"U": U, "V": V, "W": W}
