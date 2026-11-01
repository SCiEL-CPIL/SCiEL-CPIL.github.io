"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 14
    num_blocks = p // 2  # 7 blocks of 2x2
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each of 7 blocks, we have 7 Strassen terms
    rank = num_blocks * 7
    
    # U: shape (n*m, rank) = (4, rank)
    # V: shape (m*p, rank) = (28, rank)
    # W: shape (p*n, rank) = (28, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms: (a, b, c, d, e, f, g, h, c11, c12, c21, c22)
    strassen_terms = [
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1),
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1),
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0),
    ]
    
    for block in range(num_blocks):
        for t_idx, term in enumerate(strassen_terms):
            col = block * 7 + t_idx
            a_c, b_c, c_c, d_c, e_c, f_c, g_c, h_c, c11_c, c12_c, c21_c, c22_c = term
            
            # A part
            U[0][col] = a_c
            U[1][col] = b_c
            U[2][col] = c_c
            U[3][col] = d_c
            
            # B part
            V[0 * p + 2 * block][col] = e_c
            V[0 * p + 2 * block + 1][col] = f_c
            V[1 * p + 2 * block][col] = g_c
            V[1 * p + 2 * block + 1][col] = h_c
            
            # C part
            W[2 * block * n + 0][col] = c11_c
            W[(2 * block + 1) * n + 0][col] = c12_c
            W[2 * block * n + 1][col] = c21_c
            W[(2 * block + 1) * n + 1][col] = c22_c
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
