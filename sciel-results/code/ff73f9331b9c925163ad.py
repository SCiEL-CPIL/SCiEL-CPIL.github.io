"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # Part 1: A1 (2x1) * B1 (1x8) -> Rank 16
    # Part 2: A2 (2x2) * B2 (2x8) -> 4 blocks of 2x2x2 Strassen -> Rank 28
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 * B1
    # A1 is A[:, 0]
    # B1 is B[0, :]
    # C1[i, k] = A[i, 0] * B[0, k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A2 * B2
    # A2 is A[:, 1:3]
    # B2 is B[1:3, :]
    # Split B2 into 4 blocks of 2x2
    # Block m uses B rows 1,2 and cols 2m, 2m+1
    # A2 rows 0,1 correspond to A rows 0,1 and cols 1,2
    
    # Strassen 2x2x2 coefficients
    # M1 = (a+d)(e+i)
    # M2 = (c+d)e
    # M3 = a(f-i)
    # M4 = d(h-e)
    # M5 = (a+b)i
    # M6 = (c+d)(f-i)
    # M7 = (a+b)(h-e)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6
    
    # Mapping A indices: a=A[0,1], b=A[0,2], c=A[1,1], d=A[1,2]
    # Mapping B indices: e=B[1,2m], f=B[1,2m+1], h=B[2,2m], i=B[2,2m+1]
    
    for m_block in range(4):
        k0 = 2 * m_block
        k1 = 2 * m_block + 1
        
        # M1
        U[0 * m + 1][col] = 1  # a
        U[1 * m + 2][col] = 1  # d
        V[1 * p + k0][col] = 1  # e
        V[2 * p + k1][col] = 1  # i
        W[k0 * n + 0][col] = 1  # C00
        W[k1 * n + 1][col] = 1  # C11
        col += 1
        
        # M2
        U[1 * m + 1][col] = 1  # c
        U[1 * m + 2][col] = 1  # d
        V[1 * p + k0][col] = 1  # e
        W[k0 * n + 1][col] = 1  # C10
        col += 1
        
        # M3
        U[0 * m + 1][col] = 1  # a
        V[1 * p + k1][col] = 1  # f
        V[2 * p + k1][col] = -1  # -i
        W[k1 * n + 0][col] = 1  # C01
        col += 1
        
        # M4
        U[1 * m + 2][col] = 1  # d
        V[2 * p + k0][col] = 1  # h
        V[1 * p + k0][col] = -1  # -e
        W[k0 * n + 0][col] = -1  # -C00
        W[k0 * n + 1][col] = 1  # C10
        col += 1
        
        # M5
        U[0 * m + 1][col] = 1  # a
        U[0 * m + 2][col] = 1  # b
        V[2 * p + k1][col] = 1  # i
        W[k0 * n + 0][col] = 1  # C00
        W[k1 * n + 0][col] = 1  # C01
        col += 1
        
        # M6
        U[1 * m + 1][col] = 1  # c
        U[1 * m + 2][col] = 1  # d
        V[1 * p + k1][col] = 1  # f
        V[2 * p + k1][col] = -1  # -i
        W[k1 * n + 1][col] = 1  # C11
        col += 1
        
        # M7
        U[0 * m + 1][col] = 1  # a
        U[0 * m + 2][col] = 1  # b
        V[2 * p + k0][col] = 1  # h
        V[1 * p + k0][col] = -1  # -e
        W[k0 * n + 0][col] = -1  # -C00
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
