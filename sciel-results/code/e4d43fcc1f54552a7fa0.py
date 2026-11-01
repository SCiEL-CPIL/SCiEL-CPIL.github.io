"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7, using shared Strassen decomposition for 2x2 blocks
    rank = 7
    
    # Strassen terms for 2x2 matrix multiplication
    # u: coeffs for A (A00, A01, A10, A11)
    # v: coeffs for B (B00, B01, B10, B11)
    # w: coeffs for C (C00, C01, C10, C11)
    
    u_terms = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    
    v_terms = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    w_terms = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U
    for r in range(rank):
        for i in range(2):
            for j in range(2):
                U[i * m + j][r] = u_terms[r][i * 2 + j]
                
    # Fill V and W
    # B is 2x6. j in 0..1, k in 0..5.
    # C is 2x6. i in 0..1, k in 0..5.
    # We view B and C as 3 blocks of 2x2.
    # Block b (0..2) uses k indices 2b, 2b+1.
    
    for r in range(rank):
        for b in range(3):
            for j in range(2):
                for l in range(2):
                    # Global B index: j * p + (2*b + l)
                    v_idx = j * p + (2 * b + l)
                    V[v_idx][r] += v_terms[r][j * 2 + l]
                    
                    # Global C index: (2*b + l) * n + i
                    # w_terms[r] is indexed by [i, l]
                    for i in range(2):
                        w_idx = (2 * b + l) * n + i
                        W[w_idx][r] += w_terms[r][i * 2 + l]
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
