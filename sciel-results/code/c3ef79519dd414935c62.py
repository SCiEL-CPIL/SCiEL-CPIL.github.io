"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for M1..M7
    # Format: (u_coeffs: {(i,j): val}, v_coeffs: {(j,k_local): val}, w_coeffs: {(i_local, k_local): val})
    # i, j in {0,1} for A. j, k_local in {0,1} for B. i_local, k_local in {0,1} for C.
    
    strassen_terms = [
        # M1
        {
            'u': {(0,0): 1, (1,1): 1},
            'v': {(0,0): 1, (1,1): 1},
            'w': {(0,0): 1, (1,1): 1}
        },
        # M2
        {
            'u': {(1,0): 1, (1,1): 1},
            'v': {(0,0): 1},
            'w': {(1,0): 1, (1,1): -1}
        },
        # M3
        {
            'u': {(0,0): 1},
            'v': {(0,1): 1, (1,1): -1},
            'w': {(0,0): -1, (0,1): 1, (1,1): 1} # c11:-1, c12:+1, c22:+1
        },
        # M4
        {
            'u': {(1,1): 1},
            'v': {(1,0): 1, (0,0): -1},
            'w': {(0,0): 1, (1,0): 1} # c11:+1, c21:+1
        },
        # M5
        {
            'u': {(0,0): 1, (0,1): 1},
            'v': {(1,1): 1},
            'w': {(0,0): -1, (0,1): 1} # c11:-1, c12:+1
        },
        # M6
        {
            'u': {(1,0): 1, (0,0): -1},
            'v': {(0,0): 1, (0,1): 1},
            'w': {(1,1): 1} # c22:+1
        },
        # M7
        {
            'u': {(0,1): 1, (1,1): -1},
            'v': {(1,0): 1, (1,1): 1},
            'w': {(0,0): 1, (1,1): 1} # c11:+1, c22:+1
        }
    ]
    
    current_rank = 0
    
    # Part 1: A1 * B1 using Strassen for 5 blocks
    for block_idx in range(5):
        k_start = block_idx * 2
        for term in strassen_terms:
            r = current_rank
            # Fill U
            for (i, j), val in term['u'].items():
                # A1 is rows 0-1, cols 0-1 of A (which is 2x3)
                # U index: i * m + j
                U[i * m + j][r] += val
            
            # Fill V
            for (j, k_loc), val in term['v'].items():
                # B1 is rows 0-1 of B (which is 3x10)
                # k_global = k_start + k_loc
                # V index: j * p + k_global
                k_global = k_start + k_loc
                V[j * p + k_global][r] += val
            
            # Fill W
            for (i_loc, k_loc), val in term['w'].items():
                # C is 2x10.
                # i_global = i_loc (since n=2, A1 covers all rows)
                # k_global = k_start + k_loc
                # W index: k_global * n + i_global
                i_global = i_loc
                k_global = k_start + k_loc
                W[k_global * n + i_global][r] += val
            
            current_rank += 1
    
    # Part 2: A2 * B2 (Schoolbook)
    # A2 is col 2 of A (2x1). B2 is row 2 of B (1x10).
    for i in range(n):
        for k in range(p):
            r = current_rank
            # U: A[i, 2]
            U[i * m + 2][r] = 1
            # V: B[2, k]
            V[2 * p + k][r] = 1
            # W: C[i, k] -> W[k, i]
            W[k * n + i][r] = 1
            current_rank += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
