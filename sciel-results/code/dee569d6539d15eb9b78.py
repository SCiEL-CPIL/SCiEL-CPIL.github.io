"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Partition: A (2x3) = [A1(2x2) | A2(2x1)], B (3x6) = [B1(2x6); B2(1x6)]
    # C = A1*B1 + A2*B2
    # R(2,2,6) <= 3*R(2,2,2) = 3*7 = 21 (Strassen on column pairs)
    # R(2,1,6) = 12 (outer product)
    # Total rank = 33
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    
    # Part 1: R(2,2,6) via 3 Strassen blocks
    # A1 indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4 (row-major: i*m+j)
    # B1 indices: B[0,k]=k, B[1,k]=6+k (row-major: j*p+k)
    # W index for C[i,k] = k*n+i = 2k+i
    
    for block in range(3):
        k0 = block * 2
        
        # M1 = (A00+A11)*(B00+B11)
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[6+k0][col] = 1
        W[k0*2+0][col] = 1; W[(k0+1)*2+1][col] = 1
        col += 1
        
        # M2 = (A10+A11)*B00
        U[3][col] = 1; U[4][col] = 1
        V[k0][col] = 1
        W[k0*2+1][col] = 1; W[(k0+1)*2+1][col] = -1
        col += 1
        
        # M3 = A00*(B01-B11)
        U[0][col] = 1
        V[k0+1][col] = 1; V[6+k0+1][col] = -1
        W[(k0+1)*2+0][col] = 1; W[(k0+1)*2+1][col] = 1
        col += 1
        
        # M4 = A11*(B10-B00)
        U[4][col] = 1
        V[6+k0][col] = 1; V[k0][col] = -1
        W[k0*2+0][col] = 1; W[k0*2+1][col] = 1
        col += 1
        
        # M5 = (A00+A01)*B11
        U[0][col] = 1; U[1][col] = 1
        V[6+k0+1][col] = 1
        W[k0*2+0][col] = -1; W[(k0+1)*2+0][col] = 1
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[3][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k0+1][col] = 1
        W[(k0+1)*2+1][col] = 1
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[1][col] = 1; U[4][col] = -1
        V[6+k0][col] = 1; V[6+k0+1][col] = 1
        W[k0*2+0][col] = 1
        col += 1
    
    # Part 2: R(2,1,6) = 12 terms
    # A2: A[0,2]=index 2, A[1,2]=index 5
    # B2: B[2,k]=index 12+k
    for i in range(2):
        for k in range(6):
            U[i*3 + 2][col] = 1
            V[2*6 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
