"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition using Strassen for 2x2 blocks and schoolbook for the remainder
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 by 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # U indices: i*m + j
    # a: 0, b: 1, c: 2, d: 3
    # V indices: j*p + k
    # e (j=0,k=0): 0, f (j=0,k=1): 1, g (j=1,k=0): p, h (j=1,k=1): p+1
    # W indices: k*n + i
    # C00 (k=0,i=0): 0, C01 (k=1,i=0): n, C10 (k=0,i=1): 1, C11 (k=1,i=1): n+1
    
    def add_strassen_block(k0, k1, base_col):
        # k0, k1 are column indices in B
        # base_col is the starting column index in the decomposition
        
        # P1: (a+d)(e+h)
        c = base_col
        U[0][c] += 1 # a
        U[3][c] += 1 # d
        V[0 * p + k0][c] += 1 # e
        V[1 * p + k1][c] += 1 # h
        W[k0 * n + 0][c] += 1 # C00
        W[k1 * n + 1][c] += 1 # C11
        
        # P2: (c+d)e
        c = base_col + 1
        U[2][c] += 1 # c
        U[3][c] += 1 # d
        V[0 * p + k0][c] += 1 # e
        W[k0 * n + 1][c] += 1 # C10
        W[k1 * n + 1][c] -= 1 # C11
        
        # P3: a(f-h)
        c = base_col + 2
        U[0][c] += 1 # a
        V[0 * p + k1][c] += 1 # f
        V[1 * p + k1][c] -= 1 # -h
        W[k1 * n + 0][c] += 1 # C01
        W[k1 * n + 1][c] += 1 # C11
        
        # P4: d(g-e)
        c = base_col + 3
        U[3][c] += 1 # d
        V[1 * p + k0][c] += 1 # g
        V[0 * p + k0][c] -= 1 # -e
        W[k0 * n + 0][c] += 1 # C00
        W[k0 * n + 1][c] += 1 # C10
        
        # P5: (a+b)h
        c = base_col + 4
        U[0][c] += 1 # a
        U[1][c] += 1 # b
        V[1 * p + k1][c] += 1 # h
        W[k0 * n + 0][c] -= 1 # C00
        W[k1 * n + 0][c] += 1 # C01
        
        # P6: (c-a)(e+f)
        c = base_col + 5
        U[2][c] += 1 # c
        U[0][c] -= 1 # -a
        V[0 * p + k0][c] += 1 # e
        V[0 * p + k1][c] += 1 # f
        W[k1 * n + 1][c] += 1 # C11
        
        # P7: (b-d)(g+h)
        c = base_col + 6
        U[1][c] += 1 # b
        U[3][c] -= 1 # -d
        V[1 * p + k0][c] += 1 # g
        V[1 * p + k1][c] += 1 # h
        W[k0 * n + 0][c] += 1 # C00

    # Apply Strassen to 4 blocks of 2 columns
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        base_col = b * 7
        add_strassen_block(k0, k1, base_col)
        
    # Schoolbook for the last column (k=8)
    k = 8
    base_col = 28
    
    # a * B[0, k]
    c = base_col
    U[0][c] += 1 # a
    V[0 * p + k][c] += 1 # B[0, k]
    W[k * n + 0][c] += 1 # C[0, k]
    
    # b * B[1, k]
    c = base_col + 1
    U[1][c] += 1 # b
    V[1 * p + k][c] += 1 # B[1, k]
    W[k * n + 0][c] += 1 # C[0, k]
    
    # c * B[0, k]
    c = base_col + 2
    U[2][c] += 1 # c
    V[0 * p + k][c] += 1 # B[0, k]
    W[k * n + 1][c] += 1 # C[1, k]
    
    # d * B[1, k]
    c = base_col + 3
    U[3][c] += 1 # d
    V[1 * p + k][c] += 1 # B[1, k]
    W[k * n + 1][c] += 1 # C[1, k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
