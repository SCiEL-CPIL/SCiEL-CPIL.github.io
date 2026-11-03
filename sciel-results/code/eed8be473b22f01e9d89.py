"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # (U_coeffs, V_coeffs, W_coeffs)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def get_global_A(block, li, lj):
        i = li
        if block in [0, 2]:
            j = lj
        else:
            j = 2 + lj
        return i * 4 + j
        
    def get_global_B(block, li, lj):
        j_local = li
        r_local = lj
        if block == 0:
            j, r = j_local, r_local
        elif block == 1:
            j, r = 2 + j_local, r_local
        elif block == 2:
            j, r = j_local, 2 + r_local
        else:
            j, r = 2 + j_local, 2 + r_local
        return j * 4 + r
        
    def get_global_C(block, li, lj):
        i_local = li
        r_local = lj
        if block in [0, 1]:
            i, r = i_local, r_local
        else:
            i, r = i_local, 2 + r_local
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
