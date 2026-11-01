"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    num_blocks = p // 2 # 5 blocks
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block # 35

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Strassen terms for 2x2x2
    # Local indices:
    # A: (i, j) i in 0,1; j in 0,1
    # B: (j, k_loc) j in 0,1; k_loc in 0,1
    # C: (k_loc, i) k_loc in 0,1; i in 0,1 (Note: W index is k*n+i, so local is (k_loc, i))
    
    # Term: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of ((i,j), coeff)
    # V_coeffs: list of ((j,k_loc), coeff)
    # W_coeffs: list of ((k_loc,i), coeff)
    
    terms = [
        # M1 = (a00 + a11)(b00 + b11)
        # c00 += M1, c11 += M1
        {
            'U': [((0,0), 1), ((1,1), 1)],
            'V': [((0,0), 1), ((1,1), 1)],
            'W': [((0,0), 1), ((1,1), 1)] # k_loc=0,i=0 and k_loc=1,i=1
        },
        # M2 = (a10 + a11)b00
        # c10 += M2
        {
            'U': [((1,0), 1), ((1,1), 1)],
            'V': [((0,0), 1)],
            'W': [((0,1), 1)] # k_loc=0, i=1
        },
        # M3 = a00(b10 - b11)
        # c01 += M3, c11 += M3
        {
            'U': [((0,0), 1)],
            'V': [((1,0), 1), ((1,1), -1)],
            'W': [((1,0), 1), ((1,1), 1)] # k_loc=1, i=0 and k_loc=1, i=1
        },
        # M4 = a11(b01 - b00)
        # c00 += M4, c10 += M4
        {
            'U': [((1,1), 1)],
            'V': [((0,1), 1), ((0,0), -1)],
            'W': [((0,0), 1), ((0,1), 1)] # k_loc=0, i=0 and k_loc=0, i=1
        },
        # M5 = (a00 + a01)b11
        # c01 += M5, c00 -= M5
        {
            'U': [((0,0), 1), ((0,1), 1)],
            'V': [((1,1), 1)],
            'W': [((1,0), 1), ((0,0), -1)] # k_loc=1, i=0 and k_loc=0, i=0
        },
        # M6 = (a10 - a00)(b00 + b01)
        # c11 += M6
        {
            'U': [((1,0), 1), ((0,0), -1)],
            'V': [((0,0), 1), ((0,1), 1)],
            'W': [((1,1), 1)] # k_loc=1, i=1
        },
        # M7 = (a01 - a11)(b10 + b11)
        # c00 += M7
        {
            'U': [((0,1), 1), ((1,1), -1)],
            'V': [((1,0), 1), ((1,1), 1)],
            'W': [((0,0), 1)] # k_loc=0, i=0
        }
    ]

    for b in range(num_blocks):
        for r in range(rank_per_block):
            col = b * rank_per_block + r
            term = terms[r]
            
            # Fill U
            for (i, j), coeff in term['U']:
                u_idx = i * m + j
                U[u_idx][col] += coeff
                
            # Fill V
            for (j, k_loc), coeff in term['V']:
                k_global = 2 * b + k_loc
                v_idx = j * p + k_global
                V[v_idx][col] += coeff
                
            # Fill W
            for (k_loc, i), coeff in term['W']:
                k_global = 2 * b + k_loc
                w_idx = k_global * n + i
                W[w_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
