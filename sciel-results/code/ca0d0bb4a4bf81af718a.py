"""Matrix multiplication 2,2,10 - rank-35 Strassen-based decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    # Rank-35 decomposition: 5 independent (2,2,2) blocks, each using Strassen (rank 7).
    # For block b (columns 2b, 2b+1 of B), the Strassen products are:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A10)B11
    #   M6 = (A11-A00)(B00+B10)
    #   M7 = (A01-A11)(B10+B11)
    # Output: C00=M1+M4-M5-M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    #
    # Indexing: U[i*m+j] = Aij, V[j*p+k] = Bjk, W[k*n+i] = Cki
    # With n=2, m=2, p=2: U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
    #   V[0]=B00, V[1]=B01, V[2]=B10, V[3]=B11
    #   W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11

    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35

    # Strassen factor definitions for a single (2,2,2) block:
    # Each entry is a list of (row_index, coeff) for that factor
    # U factors (4 rows: A00, A01, A10, A11)
    # V factors (4 rows: B00, B01, B10, B11)
    # W factors (4 rows: C00, C01, C10, C11)
    strassen_factors = [
        # M1: (A00+A11)(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, -1)]),
        # M2: (A10+A11)B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3: A00(B01-B11)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4: A11(B10-B00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5: (A00+A10)B11
        ([(0, 1), (2, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6: (A11-A00)(B00+B10)
        ([(3, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),
        # M7: (A01-A11)(B10+B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, -1)]),
    ]

    for b in range(5):
        # Column offset for this block's B columns and C columns
        col_off = 2 * b
        for s in range(7):
            r = b * 7 + s
            u_factors, v_factors, w_factors = strassen_factors[s]
            for row, coeff in u_factors:
                U[row][r] += coeff
            for row, coeff in v_factors:
                V[row + col_off][r] += coeff
            for row, coeff in w_factors:
                W[row + col_off][r] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
