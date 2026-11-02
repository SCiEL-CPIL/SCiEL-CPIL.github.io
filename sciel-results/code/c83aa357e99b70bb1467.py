"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen decomposition for 2x2 x 2x2 blocks.
    # p=12, so 6 blocks of size 2.
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2), B (2x2), C (2x2)
    # A indices: 00, 01, 10, 11
    # B indices in block: e(0,k0), f(0,k1), g(1,k0), h(1,k1)
    # C indices in block: i(0,k0), j(0,k1), k(1,k0), l(1,k1)
    
    # u: coeffs for A
    # v: coeffs for B (e, f, g, h)
    # w: coeffs for C (i, j, k, l)
    
    strassen = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 0, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    for block in range(num_blocks):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # B indices in V
        idx_e = 0 * p + k0  # B[0, k0]
        idx_f = 0 * p + k1  # B[0, k1]
        idx_g = 1 * p + k0  # B[1, k0]
        idx_h = 1 * p + k1  # B[1, k1]
        
        # C indices in W
        # W[k*n + i]
        idx_i = k0 * n + 0  # C[0, k0]
        idx_j = k1 * n + 0  # C[0, k1]
        idx_k = k0 * n + 1  # C[1, k0]
        idx_l = k1 * n + 1  # C[1, k1]
        
        for r in range(7):
            col = block * 7 + r
            u_coeffs, v_coeffs, w_coeffs = strassen[r]
            
            # Fill U
            for i in range(4):
                U[i][col] = u_coeffs[i]
                
            # Fill V
            V[idx_e][col] = v_coeffs[0]
            V[idx_f][col] = v_coeffs[1]
            V[idx_g][col] = v_coeffs[2]
            V[idx_h][col] = v_coeffs[3]
            
            # Fill W
            W[idx_i][col] = w_coeffs[0]
            W[idx_j][col] = w_coeffs[1]
            W[idx_k][col] = w_coeffs[2]
            W[idx_l][col] = w_coeffs[3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
