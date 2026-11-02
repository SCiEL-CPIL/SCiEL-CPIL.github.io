"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Decompose m=3 as m1=1, m2=2
    # Decompose p=8 as four blocks of size 2
    m1, m2 = 1, 2
    num_p_blocks = 4
    p_block_size = 2
    
    # Part 1: (n x m1) x (m1 x p) schoolbook rank = n*m1*p = 16
    rank1 = n * m1 * p  # 16
    # Part 2: (n x m2) x (m2 x p) with Strassen on each 2x2x2 block
    rank2 = num_p_blocks * 7  # 28
    total_rank = rank1 + rank2  # 44
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: schoolbook for j=0 (the m1=1 part)
    for i in range(n):
        for k in range(p):
            j = 0
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: Strassen for each (2x2) x (2x2) block
    # A submatrix: rows i in {0,1}, cols j in {1,2} (m2 part)
    # B submatrix: rows j in {1,2}, cols k in {k_base, k_base+1}
    # C submatrix: rows i in {0,1}, cols k in {k_base, k_base+1}
    for pb in range(num_p_blocks):
        k_base = pb * p_block_size
        for s in range(7):
            if s == 0:  # M1 = (A[0][1]+A[1][2])(B[1][kb]+B[2][kb+1])
                U[0*m+1][col] = 1; U[1*m+2][col] = 1
                V[1*p+k_base][col] = 1; V[2*p+k_base+1][col] = 1
                W[k_base*n+0][col] = 1; W[(k_base+1)*n+1][col] = 1
            elif s == 1:  # M2 = (A[1][1]+A[1][2])B[1][kb]
                U[1*m+1][col] = 1; U[1*m+2][col] = 1
                V[1*p+k_base][col] = 1
                W[k_base*n+1][col] = 1; W[(k_base+1)*n+1][col] = 1
            elif s == 2:  # M3 = A[0][1](B[1][kb+1]-B[2][kb+1])
                U[0*m+1][col] = 1
                V[1*p+k_base+1][col] = 1; V[2*p+k_base+1][col] = -1
                W[(k_base+1)*n+0][col] = 1; W[(k_base+1)*n+1][col] = 1
            elif s == 3:  # M4 = A[1][2](B[2][kb]-B[1][kb])
                U[1*m+2][col] = 1
                V[2*p+k_base][col] = 1; V[1*p+k_base][col] = -1
                W[k_base*n+0][col] = 1; W[k_base*n+1][col] = 1
            elif s == 4:  # M5 = (A[0][1]+A[0][2])B[2][kb+1]
                U[0*m+1][col] = 1; U[0*m+2][col] = 1
                V[2*p+k_base+1][col] = 1
                W[k_base*n+0][col] = -1; W[(k_base+1)*n+0][col] = 1
            elif s == 5:  # M6 = (A[1][1]-A[0][1])(B[1][kb]+B[1][kb+1])
                U[1*m+1][col] = 1; U[0*m+1][col] = -1
                V[1*p+k_base][col] = 1; V[1*p+k_base+1][col] = 1
                W[(k_base+1)*n+1][col] = 1
            elif s == 6:  # M7 = (A[0][2]-A[1][2])(B[2][kb]+B[2][kb+1])
                U[0*m+2][col] = 1; U[1*m+2][col] = -1
                V[2*p+k_base][col] = 1; V[2*p+k_base+1][col] = 1
                W[k_base*n+0][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
