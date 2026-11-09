"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Strassen-based decomposition of 6x6 matrix multiplication.
    Partition 6x6 into 3x3 grid of 2x2 blocks.
    For each output block (i,j) and each k, use Strassen's rank-7 
    decomposition for 2x2 matrix multiplication.
    Total rank = 9 * 3 * 7 = 189.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 multiplication
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # m1 = (a0+a3)(b0+b3)
    # m2 = (a2+a3)b0
    # m3 = a0(b1-b3)
    # m4 = a3(b2-b0)
    # m5 = (a0+a1)b3
    # m6 = (a2-a0)(b0+b1)
    # m7 = (a1-a3)(b2+b3)
    #
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    strassen_terms = [
        # m1 = (a0+a3)(b0+b3)
        {
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}
        },
        # m2 = (a2+a3)b0
        {
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(1,0): 1, (1,1): -1}
        },
        # m3 = a0(b1-b3)
        {
            'U': {(0,0): 1},
            'V': {(0,1): 1, (1,1): -1},
            'W': {(0,1): 1, (1,1): 1}
        },
        # m4 = a3(b2-b0)
        {
            'U': {(1,1): 1},
            'V': {(1,0): 1, (0,0): -1},
            'W': {(0,0): 1, (1,0): 1}
        },
        # m5 = (a0+a1)b3
        {
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(0,0): -1, (0,1): 1}
        },
        # m6 = (a2-a0)(b0+b1)
        {
            'U': {(1,0): 1, (0,0): -1},
            'V': {(0,0): 1, (0,1): 1},
            'W': {(1,1): 1}
        },
        # m7 = (a1-a3)(b2+b3)
        {
            'U': {(0,1): 1, (1,1): -1},
            'V': {(1,0): 1, (1,1): 1},
            'W': {(0,0): 1}
        },
    ]
    
    col = 0
    for i in range(3):  # output block row
        for j in range(3):  # output block col
            for k in range(3):  # sum over k
                for term in strassen_terms:
                    # U: A block at (i, k) - rows 2i..2i+1, cols 2k..2k+1
                    for (r, c), coeff in term['U'].items():
                        row_idx = 2 * i + r
                        col_idx = 2 * k + c
                        U[row_idx * m + col_idx][col] = coeff
                    
                    # V: B block at (k, j) - rows 2k..2k+1, cols 2j..2j+1
                    for (r, c), coeff in term['V'].items():
                        row_idx = 2 * k + r
                        col_idx = 2 * j + c
                        V[row_idx * p + col_idx][col] = coeff
                    
                    # W: C block at (i, j) - output C[2i+r][2j+c]
                    # W index convention: W[k*n+i, r] means contribution to C[i][k]
                    # So W row = col_of_C * n + row_of_C
                    for (r, c), coeff in term['W'].items():
                        row_of_C = 2 * i + r
                        col_of_C = 2 * j + c
                        W[col_of_C * n + row_of_C][col] = coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
