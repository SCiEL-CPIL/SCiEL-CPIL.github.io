"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 Strassen terms = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # Format: (U_coeffs, V_coeffs, W_coeffs)
    # Keys are local indices 0..3 (00, 01, 10, 11)
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),       # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),       # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),       # M7
    ]
    
    # Block mapping
    # Block 0: A_L (cols 0,1) * B_LL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    # Block 1: A_R (cols 2,3) * B_RL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    # Block 2: A_L (cols 0,1) * B_LR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    # Block 3: A_R (cols 2,3) * B_RR (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    
    def get_global_A(block, li, lj):
        i = li
        j = lj if block in [0, 2] else 2 + lj
        return i * 4 + j
        
    def get_global_B(block, li, lj):
        j = li if block in [0, 2] else 2 + li
        r = lj if block in [0, 1] else 2 + lj
        return j * 4 + r
        
    def get_global_C(block, li, lj):
        i = li
        r = lj if block in [0, 1] else 2 + lj
        return r * 2 + i

    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = get_global_A(b, li, lj)
                U[g_idx][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = get_global_B(b, li, lj)
                V[g_idx][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = get_global_C(b, li, lj)
                W[g_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
